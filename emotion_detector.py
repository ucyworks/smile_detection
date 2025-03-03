import cv2
import numpy as np
from deepface import DeepFace
import time

def detect_emotions():
    # Kamera başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return
        
    prev_time = 0
    
    # Duygu etiketlerini Türkçeleştir
    emotion_labels = {
        'angry': 'kizgin',
        'disgust': 'igrenmis',
        'fear': 'korkmus',
        'happy': 'mutlu',
        'sad': 'uzgun',
        'surprise': 'saskin',
        'neutral': 'normal'
    }
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # FPS hesapla
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            try:
                # Duygu analizi yap
                result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                if result:
                    # En baskın duyguyu al
                    emotion = result[0]['dominant_emotion']
                    emotion_scores = result[0]['emotion']
                    
                    # Yüz konumunu al
                    face_coords = result[0].get('region', {})
                    x = face_coords.get('x', 0)
                    y = face_coords.get('y', 0)
                    w = face_coords.get('w', 0)
                    h = face_coords.get('h', 0)
                    
                    # Renk seçimi (mutluluk seviyesine göre)
                    happiness_level = emotion_scores['happy']
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
                    text = f"{emotion_labels[emotion]}: %{int(emotion_scores[emotion])}"
                    cv2.putText(frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # Tüm duygu yüzdelerini göster
                    y_offset = 30
                    for emo, score in emotion_scores.items():
                        text = f"{emotion_labels.get(emo, emo)}: %{int(score)}"
                        cv2.putText(frame, text, (10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        y_offset += 30
            
            except Exception as e:
                pass  # Analiz hatalarını sessizce geç
            
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
