# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.analyzer import analyze_context_full

app = FastAPI(title="Totem AI Context Analyzer")

# Allow local Streamlit or other hosts to call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    user_prompt: str
    ai_response: str
    output_language: str = "auto"  # 'auto' or language code like 'en', 'hi'

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = analyze_context_full(
        user_prompt=req.user_prompt,
        ai_response=req.ai_response,
        user_lang=req.output_language
    )
    return result

@app.get("/")
def root():
    return {"status": "ok", "message": "Totem AI Context Analyzer API"}
