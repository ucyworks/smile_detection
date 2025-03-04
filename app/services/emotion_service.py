import cv2
import numpy as np
from deepface import DeepFace
import base64
from typing import Dict, Any

class EmotionService:
    def __init__(self):
        # Duygu etiketlerini Türkçeleştir
        self.emotion_labels = {
            'angry': 'kizgin',
            'disgust': 'igrenmis',
            'fear': 'korkmus',
            'happy': 'mutlu',
            'sad': 'uzgun',
            'surprise': 'saskin',
            'neutral': 'normal'
        }
    
    def analyze_image(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze emotions from an image
        
        Args:
            image_data: numpy array of the image
            
        Returns:
            Dictionary with emotion data
        """
        try:
            # Duygu analizi yap
            result = DeepFace.analyze(
                image_data, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if not result:
                return {"error": "No face detected"}
                
            # En baskın duyguyu al
            emotion = result[0]['dominant_emotion']
            emotion_scores = result[0]['emotion']
            
            # Yüz konumunu al
            face_coords = result[0].get('region', {})
            x = face_coords.get('x', 0)
            y = face_coords.get('y', 0)
            w = face_coords.get('w', 0)
            h = face_coords.get('h', 0)
            
            # Sonuçları hazırla
            happiness_level = emotion_scores['happy']
            
            return {
                "dominant_emotion": {
                    "name": emotion,
                    "turkish_name": self.emotion_labels.get(emotion, emotion),
                    "score": emotion_scores[emotion]
                },
                "all_emotions": {
                    emotion: {
                        "score": score, 
                        "turkish_name": self.emotion_labels.get(emotion, emotion)
                    } for emotion, score in emotion_scores.items()
                },
                "face_location": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                },
                "happiness_level": happiness_level
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def process_base64_image(self, base64_image: str) -> Dict[str, Any]:
        """
        Process a base64 encoded image and return emotion data
        
        Args:
            base64_image: Base64 encoded image string
            
        Returns:
            Dictionary with emotion data
        """
        try:
            # Base64 görüntüyü çöz
            image_data = base64.b64decode(base64_image)
            # NumPy dizisine dönüştür
            nparr = np.frombuffer(image_data, np.uint8)
            # OpenCV görüntüsüne dönüştür
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "Invalid image data"}
                
            # Duygu analizini yap
            return self.analyze_image(image)
            
        except Exception as e:
            return {"error": f"Image processing error: {str(e)}"}
