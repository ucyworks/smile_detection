from pydantic import BaseModel
from typing import Dict, Any

class EmotionDetail(BaseModel):
    score: float
    turkish_name: str

class DominantEmotion(BaseModel):
    name: str
    turkish_name: str
    score: float

class FaceLocation(BaseModel):
    x: int
    y: int
    width: int
    height: int

class EmotionResponse(BaseModel):
    dominant_emotion: DominantEmotion
    all_emotions: Dict[str, EmotionDetail]
    face_location: FaceLocation
    happiness_level: float

class ErrorResponse(BaseModel):
    error: str
