from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    """Application settings"""
    APP_NAME: str = "Emotion Detection API"
    DEBUG: bool = False
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()
