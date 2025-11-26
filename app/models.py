# app/models.py
from pydantic import BaseModel
from typing import List, Optional

class MissingTopic(BaseModel):
    topic: str
    max_similarity: float
    confidence: float
    suggestion_en: str
    suggestion_local: str

class AnalyzeResponse(BaseModel):
    detected_user_lang: str
    output_language: str
    summary: str
    quality_score: float
    missing_topics: List[MissingTopic]
    follow_up_prompts: List[str]
    improved_answer: Optional[str]
