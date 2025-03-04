import base64
import time
import cv2
import numpy as np
from typing import Dict, Any
import os
from pathlib import Path
from fer import FER

class EmotionService:
    def __init__(self):
        # Initialize face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize FER (Facial Emotion Recognition) detector
        try:
            self.emotion_detector = FER()
            self.using_fer = True
            print("FER emotion detector initialized successfully")
        except Exception as e:
            print(f"Failed to initialize FER: {e}")
            self.using_fer = False
            
        # Fallback: try to load TensorFlow model if FER is not available
        if not self.using_fer:
            try:
                models_dir = Path(__file__).parent.parent / "models"
                self.emotion_model_path = os.path.join(models_dir, "emotion_model.h5")
                
                import tensorflow as tf
                if os.path.exists(self.emotion_model_path):
                    self.emotion_model = tf.keras.models.load_model(self.emotion_model_path)
                    self.using_tf_model = True
                    print("TensorFlow emotion model loaded successfully")
                else:
                    self.using_tf_model = False
                    print("TensorFlow emotion model file not found")
            except Exception as e:
                print(f"Failed to load TensorFlow emotion model: {e}")
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
            
            processing_time = time.time() - start_time
            
            # If using FER, detect emotions directly on the image
            if self.using_fer:
                # FER works with RGB images, OpenCV uses BGR
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = self.emotion_detector.detect_emotions(rgb_img)
                
                if result and len(result) > 0:
                    # Get the first detected face (usually the most prominent)
                    face = result[0]
                    emotions = face['emotions']
                    
                    # Find emotion with highest confidence
                    max_emotion = max(emotions.items(), key=lambda x: x[1])
                    emotion_name = max_emotion[0]
                    confidence = max_emotion[1]
                    
                    print(f"Detected emotions: {emotions}")
                    
                    return {
                        "emotion": emotion_name,
                        "confidence": float(confidence),
                        "processing_time": processing_time
                    }
                else:
                    return {
                        "emotion": "no face detected",
                        "confidence": 0.0,
                        "processing_time": processing_time
                    }
            else:
                # Fall back to our custom detection method
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
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
            print(f"Error processing image: {str(e)}")
            return {"error": f"Image processing error: {str(e)}"}
    
    def _detect_emotion(self, face_roi):
        """Custom emotion detection as fallback when FER is not available"""
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
                
                print(f"TF model predictions: {list(zip(emotions, predictions[0]))}")
                
                return emotion, confidence
            else:
                # Advanced fallback using various facial feature detectors
                # This approach uses multiple cascades to detect different expressions
                smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                
                # Resize for consistent detection
                resized_face = cv2.resize(face_roi, (200, 200))
                
                # Dictionary to store emotion confidences
                emotions = {
                    'happy': 0.0,
                    'sad': 0.0,
                    'angry': 0.0,
                    'surprise': 0.0,
                    'neutral': 0.0
                }
                
                # Detect smiles - indicator of happiness
                smiles = smile_cascade.detectMultiScale(
                    resized_face,
                    scaleFactor=1.8,
                    minNeighbors=15,
                    minSize=(25, 25)
                )
                
                if len(smiles) > 0:
                    # Calculate happiness confidence based on smile size
                    largest_smile = max(smiles, key=lambda rect: rect[2] * rect[3])
                    smile_area = largest_smile[2] * largest_smile[3]
                    face_area = 200 * 200
                    emotions['happy'] = min(0.9, smile_area / (face_area * 0.3))
                
                # Detect eyes - useful for various emotions
                eyes = eye_cascade.detectMultiScale(resized_face)
                
                # Eye detection for surprise (wide open eyes)
                if len(eyes) >= 2:
                    # Calculate average eye size
                    avg_eye_area = sum(e[2] * e[3] for e in eyes) / len(eyes)
                    face_area = 200 * 200
                    
                    # Larger eyes might indicate surprise
                    if avg_eye_area > (face_area * 0.02):
                        emotions['surprise'] = min(0.7, avg_eye_area / (face_area * 0.05))
                
                # Analyze image statistics for other emotions
                brightness = np.mean(resized_face)
                contrast = np.std(resized_face)
                
                # Lower brightness might correlate with negative emotions
                if brightness < 100:
                    emotions['sad'] = 0.5 + (100 - brightness) / 200
                    emotions['angry'] = 0.3 + (100 - brightness) / 300
                
                # Higher contrast might indicate stronger expressions
                if contrast > 50:
                    # Increase all emotions slightly based on contrast
                    for emotion in emotions:
                        if emotions[emotion] > 0:
                            emotions[emotion] = min(0.95, emotions[emotion] + (contrast - 50) / 200)
                
                # If no strong emotions detected, default to neutral
                if all(conf < 0.4 for conf in emotions.values()):
                    emotions['neutral'] = 0.7
                
                # Find the dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                
                print(f"Custom detection emotions: {emotions}")
                
                return dominant_emotion[0], dominant_emotion[1]
            
        except Exception as e:
            print(f"Error in _detect_emotion: {e}")
            # Return neutral as fallback
            return "neutral", 0.5
