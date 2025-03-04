from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    # API settings
    APP_NAME: str = "Emotion Detection API"
    DEBUG: bool = False
    PORT: int = 8000
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Other settings can be added as needed
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()
