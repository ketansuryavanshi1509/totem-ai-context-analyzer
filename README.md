
# Totem — AI Context Analyzer (Prototype)

Totem is an AI-driven tool that evaluates the **quality**, **completeness**, and **semantic coverage** of an AI-generated answer based on a user’s original question.  
It detects **missing topics**, assigns a **quality score**, generates **follow-up prompts**, and produces **localized meta-guidance** on how to improve the answer.

It supports **multiple languages** (English, Hindi, Marathi, Spanish) and uses multilingual embeddings for semantic comparison.

---

## 🚀 Features

### 🔍 1. Semantic Coverage Detection
- Splits the user prompt into meaningful sentences  
- Embeds each sentence using multilingual MiniLM  
- Scores how well the AI’s answer covers each part of the question

### 📊 2. Quality Score (0–10)
- Based on semantic similarity  
- Penalizes answers that are:
  - Too short  
  - Off-topic  
  - Missing parts of the question

### ⚠️ 3. Missing Topic Extraction
Identifies which parts of the query are **not answered at all**.

### 💬 4. Follow-up Prompt Generation
Generates intelligent follow-up questions to get a better answer:
- Ask for examples
- Ask for clarity
- Ask for missing topics
- Ask for limitations / steps

### 🧠 5. Improved Answer (Meta-Guidance)
A non-generative module that:
- Analyzes what is missing  
- Suggests what to add  
- Gives improvement structure  
- Does **NOT** hallucinate or fabricate facts  
(Preferred for evaluation, safe & deterministic.)

### 🌐 6. Multilingual Support
Works in:
- English (`en`)
- Hindi (`hi`)
- Marathi (`mr`)
- Spanish (`es`)

Localized output for:
- Summary  
- Follow-up prompts  
- Suggestions  
- Improvement guidance  

### 🖥 7. Streamlit UI
A simple, clean UI for testing:

- Paste user prompt  
- Paste AI response  
- Choose language  
- View score, missing topics, suggestions, guidance

---

## 📁 Project Structure

```

totem-assignment/
├─ app/
│  ├─ main.py              # FastAPI backend
│  ├─ analyzer.py          # Core logic (scoring, embeddings, multilingual output)
│  ├─ models.py            # Request/Response schemas
│  └─ utils/
│     ├─ langutils.py      # Language detection + templates
│     └─ textutils.py      # Splitting + cleaning
│
├─ ui/
│  └─ streamlit_app.py     # Frontend UI
│
├─ samples/
│  └─ sample_input.json
│
├─ requirements.txt
└─ README.md


## 🔧 Tech Stack

- **FastAPI** (backend)
- **Streamlit** (frontend)
- **Sentence-Transformers**  
  - Model: `paraphrase-multilingual-MiniLM-L12-v2`
- **PyTorch**
- **langdetect**
- **NumPy**

No training required.  
All models run in inference mode.

---

## 🛠 Installation & Setup

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
````

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run Backend (FastAPI)

```bash
uvicorn app.main:app --reload --port 8000
```

API Docs:
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 4️⃣ Run Frontend (Streamlit)

In a new terminal:

```bash
streamlit run ui/streamlit_app.py
```

UI opens at:
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧠 How It Works (Logic Flow)

### **1. Sentence Splitting**

User prompt is split and cleaned.

### **2. Instruction Filtering**

Generic instructions like:

* "Explain with example"
* "Give example"
* "Explain step by step"

are **ignored** in scoring.

### **3. Embedding**

The system generates embeddings for:

* each user sentence
* each AI answer sentence

### **4. Semantic Similarity**

Cosine similarity is computed.

If similarity < threshold → marked as **missing topic**.

### **5. Quality Score**

Average similarity → scaled to 0–10.

### **6. Multilingual Output**

Localized strings generated using predefined templates.

### **7. Follow-up Prompts**

Generated based on what's missing.

### **8. Improved Answer (Meta-Guidance)**

Outlines how to fix the shortcomings without generating new factual content.

---

## 🧪 Example Tests

### 🔴 **Bad Answer Example**

**User Prompt:**

> What is machine learning? Explain with an example.

**AI Response:**

> Machine learning is a dance.

➡ Score: **Very low**
➡ Missing topic: full explanation, example
➡ Follow-up prompts suggested
➡ Guidance given

---

### 🟢 **Good Answer Example**

**User Prompt:**

> What is overfitting in machine learning? Explain with an example.

**AI Response:**

> Overfitting happens when...

➡ Score: **High**
➡ Minimal missing topics
➡ Better guidance

---

### 🌐 **Hindi Example**

Prompt:

```
कृत्रिम बुद्धिमत्ता क्या है? इसके प्रकार और उपयोग भी बताएं।
```

➡ System detects `hi`
➡ Output localized in Hindi

---

## 📌 Limitations

* Does not check factual correctness
* Does not generate new answers (only meta-feedback)
* Multilingual output uses templates, not full translation
* Large questions with many subtopics may need fine-tuning

---

## 🚀 Future Improvements

* Add small generative model (optional)
* Add deeper translation models
* Add context history support
* Add weighted coverage scoring
* UI improvements

---

## 👤 Author

ketan suryavanshi
AI Developer | Backend Developer

---

## 📜 License

This project is for educational and assignment purposes.

