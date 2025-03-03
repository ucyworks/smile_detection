import cv2
import numpy as np
from deepface import DeepFace
import base64
import time
from typing import Dict, Any, Optional, Tuple

class EmotionDetector:
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

# Test için yerel çalıştırma
def detect_emotions():
    detector = EmotionDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return
        
    prev_time = 0
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # FPS hesapla
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            result = detector.analyze_image(frame)
            
            if "error" not in result:
                # Yüz konumunu al
                face_loc = result["face_location"]
                x, y, w, h = face_loc["x"], face_loc["y"], face_loc["width"], face_loc["height"]
                
                # Renk seçimi (mutluluk seviyesine göre)
                happiness_level = result["happiness_level"]
                if happiness_level > 80:
                    color = (0, 255, 255)  # Sarı (çok mutlu)
                elif happiness_level > 50:
                    color = (0, 255, 0)    # Yeşil (mutlu)
                elif happiness_level > 20:
                    color = (0, 165, 255)  # Turuncu (az mutlu)
                else:
                    color = (0, 0, 255)    # Kırmızı (mutsuz)
                
                # Yüz çerçevesini çiz
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Duygu durumunu ve yüzdesini yaz
                dominant = result["dominant_emotion"]
                text = f"{dominant['turkish_name']}: %{int(dominant['score'])}"
                cv2.putText(frame, text, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Tüm duygu yüzdelerini göster
                y_offset = 30
                for emotion, data in result["all_emotions"].items():
                    text = f"{data['turkish_name']}: %{int(data['score'])}"
                    cv2.putText(frame, text, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_offset += 30
            
            # FPS göster
            cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Duygu Analizi", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Hata oluştu: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions()
