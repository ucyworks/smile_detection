from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import logging
import time

from emotion_detector import EmotionDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize the app
app = FastAPI(
    title="Emotion Detection API",
    description="API for detecting emotions from images",
    version="1.0.0+1",
)

# CORS middleware for allowing cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the emotion detector
emotion_detector = EmotionDetector()

# Request model for emotion detection
class EmotionDetectionRequest(BaseModel):
    image: str  # base64 encoded image
    language: Optional[str] = "tr"  # Default to Turkish

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def root():
    return {"message": "Welcome to Emotion Detection API. Use /detect-emotion endpoint to analyze images."}

@app.post("/detect-emotion")
async def detect_emotion(request: EmotionDetectionRequest):
    """
    Detect emotions in a base64 encoded image
    """
    try:
        logger.info("Processing emotion detection request")
        result = emotion_detector.process_base64_image(request.image)
        
        if "error" in result:
            logger.error(f"Error in emotion detection: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
            
        logger.info("Emotion detection successful")
        return result
    except Exception as e:
        logger.exception("Unexpected error in emotion detection")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "detail": str(exc)},
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
