# Totem AI Context Analyzer — Assignment Solution

This repository contains a minimal working prototype for the Totem AI Developer assignment:
- analyzes a user's prompt and an AI's reply,
- identifies missing details/gaps,
- generates follow-up prompts and confidence scores,
- provides a short summary and quality score,
- optionally generates an improved answer,
- supports multilingual analysis using multilingual sentence embeddings.

## How it works (short)
1. Sentences are extracted from the user prompt and AI response.
2. Multilingual sentence-transformer embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) are computed.
3. Each user sentence is compared to AI sentences using cosine similarity.
4. If similarity < threshold (0.55), that user sentence is considered “missing” and a follow-up prompt is generated.
5. A simple quality score is computed from average similarity (0..10).
6. Optionally, an improved answer is generated using `flan-t5-small` (if installed).

## Run locally (recommended)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# start backend
uvicorn app.main:app --reload --port 8000

# in a second terminal, start UI
streamlit run ui/streamlit_app.py
