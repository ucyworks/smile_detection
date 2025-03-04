from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.models import ImageRequest, EmotionResponse
from app.services import EmotionService
from app.config import get_settings

app = FastAPI(
    title="Emotion Detection API",
    description="API for detecting emotions from images",
    version="1.0.0"
)

settings = get_settings()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

emotion_service = EmotionService()

@app.get("/")
async def root():
    return {"message": "Welcome to Emotion Detection API. Visit /docs for documentation."}

@app.post("/detect-emotion", response_model=EmotionResponse)
async def detect_emotion(request: ImageRequest):
    try:
        # Process the base64 image
        result = emotion_service.process_base64_image(request.image)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)
