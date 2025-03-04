from pydantic import BaseModel, Field, validator
import re

class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    
    @validator('image')
    def validate_base64(cls, v):
        if not v:
            raise ValueError('Empty base64 string')
        
        # Check if it's a valid base64 string (simple check)
        base64_pattern = r'^[A-Za-z0-9+/]+={0,2}$'
        if not re.match(base64_pattern, v):
            raise ValueError('Invalid base64 encoding')
            
        return v
