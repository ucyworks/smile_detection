import requests
import base64
import json
import cv2
import os

def capture_image():
    """Capture an image from webcam and convert to base64"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam could not be accessed")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not capture image")
        return None
    
    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return image_base64

def send_to_api(image_base64, api_url="http://localhost:8000/detect-emotion"):
    """Send image to API and print response"""
    headers = {
        'Content-Type': 'application/json'
    }
    
    payload = {
        'image': image_base64,
        'language': 'tr'
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("API Response:")
        print(json.dumps(result, indent=2))
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error sending request to API: {e}")
        return None

def main():
    print("Capturing image from webcam...")
    image_base64 = capture_image()
    
    if image_base64:
        print("Sending to API...")
        send_to_api(image_base64)
    else:
        print("Failed to capture image.")

if __name__ == "__main__":
    main()
