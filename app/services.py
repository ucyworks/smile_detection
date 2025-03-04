import base64
import time
import cv2
import numpy as np
from typing import Dict, Any
import os
from pathlib import Path

class EmotionService:
    def __init__(self):
        # Initialize face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load pre-trained emotion recognition model if available
        models_dir = Path(__file__).parent.parent / "models"
        
        # Initialize emotion detection model
        try:
            # Try loading a pre-trained model - here we're using FER (Facial Emotion Recognition) model
            # You may need to download and save these files in your models directory
            self.emotion_model_path = os.path.join(models_dir, "emotion_model.h5")
            
            # If you use TensorFlow/Keras model:
            # Try importing TensorFlow first
            try:
                import tensorflow as tf
                if os.path.exists(self.emotion_model_path):
                    self.emotion_model = tf.keras.models.load_model(self.emotion_model_path)
                    self.using_tf_model = True
                else:
                    self.using_tf_model = False
            except ImportError:
                self.using_tf_model = False
                
        except Exception as e:
            print(f"Failed to load emotion model: {e}")
            self.using_tf_model = False
    
    def process_base64_image(self, base64_image: str) -> Dict[str, Any]:
        """Process a base64 encoded image and detect emotions"""
        start_time = time.time()
        
        try:
            # Remove data:image/jpeg;base64, if present
            if "," in base64_image:
                base64_image = base64_image.split(",")[1]
                
            # Decode the base64 string
            img_data = base64.b64decode(base64_image)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"error": "Invalid image data"}
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            processing_time = time.time() - start_time
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                
                # Use emotion detection logic
                emotion, confidence = self._detect_emotion(face_roi)
                
                return {
                    "emotion": emotion, 
                    "confidence": confidence, 
                    "processing_time": processing_time
                }
            else:
                return {
                    "emotion": "no face detected",
                    "confidence": 0.0,
                    "processing_time": processing_time
                }
                
        except Exception as e:
            return {"error": f"Image processing error: {str(e)}"}
    
    def _detect_emotion(self, face_roi):
        """Detect emotion from face image"""
        try:
            if self.using_tf_model:
                # If TensorFlow model is available, use it
                import tensorflow as tf
                
                # Preprocess for the model
                face_img = cv2.resize(face_roi, (48, 48))
                face_img = face_img / 255.0
                face_img = np.reshape(face_img, (1, 48, 48, 1))
                
                # Predict emotion
                predictions = self.emotion_model.predict(face_img)
                emotion_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][emotion_idx])
                
                # Map to emotion labels
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                emotion = emotions[emotion_idx]
            else:
                # Fallback to simple smile detection if no model is available
                smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
                
                # Resize for better detection
                resized_face = cv2.resize(face_roi, (250, 250))
                
                # Detect smiles
                smiles = smile_cascade.detectMultiScale(
                    resized_face,
                    scaleFactor=1.7,
                    minNeighbors=22,
                    minSize=(25, 25)
                )
                
                # Calculate confidence based on smile detection results
                if len(smiles) > 0:
                    # If smiles detected, calculate confidence based on size relative to face
                    largest_smile = max(smiles, key=lambda rect: rect[2] * rect[3])
                    smile_area = largest_smile[2] * largest_smile[3]
                    face_area = 250 * 250
                    confidence = min(0.95, smile_area / (face_area * 0.5))
                    emotion = "happy"
                else:
                    # If no smile detected, check for other features
                    # This is a simplified approach; a real model would be more nuanced
                    # Here we use simple histogram analysis as a rough proxy
                    hist = cv2.calcHist([resized_face], [0], None, [256], [0, 256])
                    brightness = np.mean(resized_face)
                    
                    if brightness < 100:  # Darker expression
                        emotion = "sad"
                        confidence = 0.6
                    else:
                        emotion = "neutral"
                        confidence = 0.7
            
            return emotion, float(confidence)
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            # Return neutral as fallback
            return "neutral", 0.5
