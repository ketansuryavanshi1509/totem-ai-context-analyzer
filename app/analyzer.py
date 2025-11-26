# app/analyzer.py

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, util

from app.utils.langutils import detect_language, translate_text_if_needed  # translate is a no-op stub now
from app.utils.textutils import split_sentences, clean_text

# Multilingual sentence embedding model
EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMB = SentenceTransformer(EMB_MODEL_NAME)

SIM_THRESHOLD = 0.55   # below this => consider that user sentence not covered
MIN_SENT_LEN = 3       # ignore very short fragments
MIN_AI_WORDS = 15      # if AI answer is too short, treat as low-quality


# Sentences that are just instructions like "Explain with an example"
INSTRUCTION_FRAGMENTS = [
    "explain with an example",
    "give an example",
    "with an example",
    "with examples",
    "give two examples",
    "give two real-life examples",
    "explain in detail",
    "explain step by step",
]


# ------------------ Localization helpers ------------------ #

def normalize_lang(lang: str) -> str:
    """
    Normalize language code to one of: en, hi, mr, es
    """
    if not lang:
        return "en"
    lang = lang.lower()
    if lang.startswith("hi"):
        return "hi"
    if lang.startswith("mr"):
        return "mr"
    if lang.startswith("es"):
        return "es"
    return "en"


def make_suggestion(topic: str, lang: str) -> str:
    """
    Localized suggestion text for a missing topic.
    """
    if lang == "hi":
        return f"कृपया इस प्रश्न के लिए सही और विस्तृत उत्तर दीजिए: '{topic}'. सरल व्याख्या और एक उदाहरण भी जोड़ें।"
    elif lang == "mr":
        return f"कृपया या प्रश्नाचे योग्य आणि सविस्तर उत्तर द्या: '{topic}'. सोपी समजावणी आणि एक उदाहरण जोडा."
    elif lang == "es":
        return f"Por favor da una respuesta completa para: '{topic}'. Incluye una explicación sencilla y un ejemplo."
    else:  # English
        return f"Please give a proper, detailed answer for: '{topic}'. Provide explanation and an example."


def make_generic_followups(lang: str, has_missing: bool) -> List[str]:
    """
    Localized generic follow-up prompts.
    """
    if lang == "hi":
        if not has_missing:
            return [
                "आपका उत्तर लगभग पूरा है। क्या आप कोई वास्तविक उदाहरण जोड़ सकते हैं?",
                "कृपया मुख्य बिंदुओं को सरल भाषा में 2–3 वाक्यों में समझाइए।",
            ]
        else:
            return [
                "कृपया ऊपर दिए गए बिंदुओं के लिए विस्तार से उत्तर लिखिए।",
                "स्पष्टता के लिए एक व्यावहारिक उदाहरण भी दीजिए।",
                "यदि कोई सीमाएँ या खास केस हैं तो उन्हें भी समझाइए।",
            ]
    elif lang == "mr":
        if not has_missing:
            return [
                "तुमचे उत्तर जवळजवळ पूर्ण आहे. एखादे प्रत्यक्ष उदाहरण देऊ शकता का?",
                "कृपया मुख्य मुद्दे २–३ सोप्या वाक्यांत समजावून सांगा.",
            ]
        else:
            return [
                "वरील मुद्द्यांवर अधिक सविस्तर उत्तर लिहा.",
                "स्पष्टतेसाठी एखादे प्रत्यक्ष उदाहरण द्या.",
                "महत्वाच्या मर्यादा किंवा विशेष केसेस असल्यास त्या देखील नमूद करा.",
            ]
    elif lang == "es":
        if not has_missing:
            return [
                "Tu respuesta está casi completa. ¿Puedes añadir un ejemplo real?",
                "Explica las ideas principales en 2–3 frases sencillas.",
            ]
        else:
            return [
                "Por favor desarrolla más los puntos indicados arriba.",
                "Añade un ejemplo práctico para que sea más claro.",
                "Si hay limitaciones o casos especiales, explícalos también.",
            ]
    else:  # English
        if not has_missing:
            return [
                "Your answer looks mostly complete. Could you provide a concrete real-world example?",
                "Could you break down the key ideas in 2–3 simple sentences?",
            ]
        else:
            return [
                "Please expand on the missing points listed above.",
                "Could you also add a practical example to make it clearer?",
                "Please explain any important limitations or edge cases as well.",
            ]


def make_summary(quality_score: float, too_short: bool, lang: str) -> str:
    """
    Localized summary of answer quality.
    """
    if too_short:
        if lang == "hi":
            return "AI उत्तर बहुत छोटा है या पर्याप्त जानकारी नहीं देता।"
        if lang == "mr":
            return "AI चे उत्तर खूप लहान आहे किंवा पुरेशी माहिती देत नाही."
        if lang == "es":
            return "La respuesta de la IA es demasiado corta o poco informativa."
        return "AI response is too short or not informative enough."

    if quality_score < 6:
        if lang == "hi":
            return "AI उत्तर में प्रश्न के कई हिस्से छूट गए हैं।"
        if lang == "mr":
            return "AI उत्तरात प्रश्नातील काही महत्वाचे भाग राहिले आहेत."
        if lang == "es":
            return "La respuesta de la IA omite varios aspectos importantes de la pregunta."
        return "AI response misses several aspects of the user's prompt."
    else:
        if lang == "hi":
            return "AI उत्तर अधिकांश बिंदु कवर करता है, लेकिन और उदाहरण या गहराई जोड़ी जा सकती है."
        if lang == "mr":
            return "AI उत्तर मुख्य मुद्दे कव्हर करतो, पण अधिक उदाहरणे किंवा सविस्तर स्पष्टीकरण देता येईल."
        if lang == "es":
            return "La respuesta de la IA cubre la mayor parte, pero puede mejorarse con más ejemplos o detalle."
        return "AI response reasonably covers the user's prompt but could use more examples or depth."


# ------------------ Embeddings + core logic helpers ------------------ #

def compute_embeddings(sentences: List[str]):
    if not sentences:
        return None
    return EMB.encode(sentences, convert_to_tensor=True, show_progress_bar=False)


def build_improved_answer(user_prompt: str, ai_response: str, missing_topics: List[Dict]) -> str:
    """
    Base (English) meta-level improved-answer description.
    This does NOT try to generate the factual answer, only guidance.
    """
    parts = []

    if ai_response.strip():
        parts.append("Current answer:\n" + ai_response.strip())
    else:
        parts.append("Current answer: (no answer provided)")

    if missing_topics:
        parts.append("\nThe answer is incomplete. It should also cover:")
        for m in missing_topics:
            parts.append(f"- {m['topic']}")

        parts.append(
            "\nHow to improve this answer:\n"
            "Start with a clear and correct definition that directly answers the question.\n"
            "Explain the key ideas or types in 2–3 short sentences.\n"
            "Add at least one real-world example so that a beginner can understand.\n"
            "Mention any important differences, pros/cons, or limitations if relevant."
        )
    else:
        parts.append(
            "\nThe answer is mostly on topic, but you can improve it by:\n"
            "Giving a more detailed explanation in simple language.\n"
            "Adding a concrete real-world example.\n"
            "Breaking down the concept into 2–3 key points."
        )

    return "\n".join(parts).strip()


def build_improved_answer_local(
    user_prompt: str,
    ai_response: str,
    missing_topics: List[Dict],
    lang: str
) -> str:
    """
    Localized wrapper around the base English meta-answer.
    Keeps base structure, adds localized guidance.
    """
    base_en = build_improved_answer(user_prompt, ai_response, missing_topics)

    if lang == "hi":
        extra = (
            "\n\nउत्तर को बेहतर बनाने के लिए:\n"
            "1. प्रश्न का सही और स्पष्ट परिभाषा से शुरुआत करें।\n"
            "2. 2–3 सरल वाक्यों में मुख्य बिंदु समझाएँ.\n"
            "3. एक वास्तविक जीवन का उदाहरण जोड़ें ताकि शुरुआत करने वाला भी समझ सके.\n"
            "4. जहाँ ज़रूरी हो, अंतर, फायदे–नुकसान या सीमाएँ भी लिखें."
        )
    elif lang == "mr":
        extra = (
            "\n\nहे उत्तर सुधारण्यासाठी:\n"
            "1. प्रश्नाचे स्पष्ट आणि बरोबर परिभाषा लिहा.\n"
            "2. मुख्य मुद्दे २–३ सोप्या वाक्यांत समजावून सांगा.\n"
            "3. सुरुवातीच्या विद्यार्थ्यालाही समजेल असा प्रत्यक्ष जीवनातील उदाहरण जोडा.\n"
            "4. गरज असेल तिथे फरक, फायदे–तोटे आणि मर्यादा नमूद करा."
        )
    elif lang == "es":
        extra = (
            "\n\nPara mejorar esta respuesta:\n"
            "1. Empieza con una definición clara y correcta.\n"
            "2. Explica las ideas clave en 2–3 frases sencillas.\n"
            "3. Añade un ejemplo del mundo real que cualquiera pueda entender.\n"
            "4. Menciona diferencias, ventajas/desventajas o limitaciones si aplican."
        )
    else:
        # English: base_en is already fine
        return base_en

    return base_en + extra


# ------------------ Main analyzer function ------------------ #

def analyze_context_full(user_prompt: str, ai_response: str, user_lang: str = "auto") -> Dict:
    """
    Core analyzer:
    - Detect user language
    - Split prompt/response into sentences
    - Filter out generic instruction-only sentences (e.g. "Explain with an example")
    - Compute similarities to find missing topics
    - Penalise very short AI answers
    - Build localized summary, follow-up prompts, and improved-answer guidance
    """

    detected_user_lang = detect_language(user_prompt) or "en"
    output_lang_raw = user_lang if user_lang not in ("", "auto", None) else detected_user_lang
    output_lang = normalize_lang(output_lang_raw)

    # Sentence splitting + cleaning
    raw_user_sentences = [
        clean_text(s) for s in split_sentences(user_prompt) if len(clean_text(s)) >= MIN_SENT_LEN
    ]

    # Filter out generic instruction sentences like "Explain with an example"
    user_sentences: List[str] = []
    for s in raw_user_sentences:
        low = s.lower()
        if any(fragment in low for fragment in INSTRUCTION_FRAGMENTS):
            continue
        user_sentences.append(s)

    # Fallback: if filtering removed everything, use original sentences
    if not user_sentences:
        user_sentences = raw_user_sentences

    ai_sentences = [
        clean_text(s) for s in split_sentences(ai_response) if len(clean_text(s)) >= MIN_SENT_LEN
    ]

    if not user_sentences:
        return {
            "detected_user_lang": detected_user_lang,
            "output_language": output_lang,
            "summary": "",
            "quality_score": 0.0,
            "missing_topics": [],
            "follow_up_prompts": ["Please provide a more detailed user prompt."],
            "improved_answer": ""
        }

    # Quick low-quality check
    ai_word_count = len(ai_response.split())
    too_short_ai = ai_word_count < MIN_AI_WORDS

    user_embs = compute_embeddings(user_sentences)
    ai_embs   = compute_embeddings(ai_sentences) if ai_sentences and not too_short_ai else None

    missing: List[Dict] = []
    followups: List[str] = []

    # ---- Gap detection ----
    if too_short_ai or not ai_sentences:
        # Treat everything as missing
        for u in user_sentences:
            suggestion = make_suggestion(u, output_lang)
            missing.append({
                "topic": u,
                "max_similarity": 0.0,
                "confidence": 1.0,
                "suggestion_en": suggestion,
                "suggestion_local": suggestion,
            })
            followups.append(suggestion)
    else:
        # Similarity-based matching user -> AI
        for idx, u in enumerate(user_sentences):
            sims = util.cos_sim(user_embs[idx], ai_embs).cpu().numpy()
            max_sim = float(np.max(sims))
            confidence = round(1.0 - max_sim, 3)

            if max_sim < SIM_THRESHOLD:
                suggestion = make_suggestion(u, output_lang)
                missing.append({
                    "topic": u,
                    "max_similarity": round(max_sim, 3),
                    "confidence": confidence,
                    "suggestion_en": suggestion,
                    "suggestion_local": suggestion,
                })
                followups.append(suggestion)

    has_missing = len(missing) > 0

    # ---- Follow-ups ----
    if not followups:
        # nothing missing => generic depth prompts
        followups = make_generic_followups(output_lang, has_missing=False)
    else:
        # add generic depth prompts even when missing exists
        extra = make_generic_followups(output_lang, has_missing=True)
        followups.extend(extra)

    # ---- Quality score & summary ----
    if too_short_ai or not ai_sentences:
        quality_score = 2.0
    else:
        sims_matrix = util.cos_sim(user_embs, ai_embs).cpu().numpy()
        max_per_user = np.max(sims_matrix, axis=1) if sims_matrix.size else np.array([0.0])
        avg_sim = float(np.mean(max_per_user))
        quality_score = round(max(0.0, min(1.0, avg_sim)) * 10.0, 2)

    summary = make_summary(quality_score, too_short_ai or not ai_sentences, output_lang)

    # ---- Improved answer (meta guidance, localized) ----
    improved_local = build_improved_answer_local(user_prompt, ai_response, missing, output_lang)

    return {
        "detected_user_lang": detected_user_lang,
        "output_language": output_lang,
        "summary": summary,
        "quality_score": quality_score,
        "missing_topics": missing,
        "follow_up_prompts": followups,
        "improved_answer": improved_local,
    }
