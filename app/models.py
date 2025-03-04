from pydantic import BaseModel, Field

class ImageRequest(BaseModel):
    """Request model for image processing"""
    image: str = Field(..., description="Base64 encoded image")

class EmotionResponse(BaseModel):
    """Response model for emotion detection"""
    emotion: str = Field(..., description="Detected emotion")
    confidence: float = Field(..., description="Confidence score")
    processing_time: float = Field(..., description="Time taken to process in seconds")
