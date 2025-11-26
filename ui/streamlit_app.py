# ui/streamlit_app.py
import streamlit as st
import requests
import json

API_URL = st.sidebar.text_input("Backend URL", value="http://localhost:8000/analyze")

st.title("Totem — AI Context Analyzer (Prototype)")
st.markdown("Enter a user prompt and the AI response. The tool will find gaps and suggest follow-up prompts.")

user_prompt = st.text_area("User Prompt", height=150, placeholder="Enter the user's original prompt here...")
ai_response = st.text_area("AI Response", height=150, placeholder="Enter the AI's response (or mock it) here...")
# 🔹 UPDATED: allow en, mr, hi, es
out_lang = st.selectbox(
    "Output language (for suggestions)",
    ["auto", "en", "mr", "hi", "es"],
    index=0,
    format_func=lambda code: {
        "auto": "Auto (use detected language)",
        "en": "English",
        "mr": "Marathi",
        "hi": "Hindi",
        "es": "Spanish",
    }.get(code, code),
)

if st.button("Analyze"):
    if not user_prompt.strip():
        st.error("Please enter a user prompt.")
    else:
        payload = {"user_prompt": user_prompt, "ai_response": ai_response, "output_language": out_lang}
        try:
            res = requests.post(API_URL, json=payload, timeout=120)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

        st.subheader("Summary & Quality")
        st.write(f"**Detected User Language:** {data.get('detected_user_lang')}")
        st.write(f"**Output Language:** {data.get('output_language')}")
        st.write(f"**Quality Score:** {data.get('quality_score')}/10")
        st.write(f"**Summary:** {data.get('summary')}")

        st.subheader("Missing Topics (detected)")
        missing = data.get("missing_topics", [])
        if not missing:
            st.write("No significant missing topics detected. Consider asking for examples or depth.")
        else:
            for m in missing:
                st.markdown(f"- **Topic:** {m['topic']}")
                st.markdown(f"  - max_similarity: {m['max_similarity']}, confidence: {m['confidence']}")
                st.markdown(f"  - Suggestion (EN): {m['suggestion_en']}")
                st.markdown(f"  - Suggestion (Local): {m.get('suggestion_local','-')}")

        st.subheader("Follow-up Prompts")
        for p in data.get("follow_up_prompts", []):
            st.write(f"- {p}")

        st.subheader("Improved Answer (auto-generated)")
        st.write(data.get("improved_answer", ""))
