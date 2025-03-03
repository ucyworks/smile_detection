# Emotion Detection API

This project provides an API for detecting emotions from images using DeepFace and OpenCV.

## Features

- Detect emotions in images via REST API
- Base64 encoded image input
- Detailed emotion analysis in response
- Ready for deployment to Railway.app
- Dockerized for easy deployment

## Prerequisites

- Python 3.8+
- Docker (optional)
- Railway CLI (optional, for deployment)

## Local Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the API:
```bash
uvicorn app:app --reload
```

The API will be available at http://localhost:8000

## Using Docker

Build the Docker image:
```bash
docker build -t emotion-detection-api .
```

Run the container:
```bash
docker run -p 8000:8000 emotion-detection-api
```

## API Endpoints

### POST /detect-emotion

Detects emotions in a base64 encoded image.

**Request Body:**
```json
{
  "image": "base64_encoded_image_string",
  "language": "tr"
}
```

**Response:**
```json
{
  "dominant_emotion": {
    "name": "happy",
    "turkish_name": "mutlu",
    "score": 95.2
  },
  "all_emotions": {
    "happy": {
      "score": 95.2,
      "turkish_name": "mutlu"
    },
    "angry": {
      "score": 0.5,
      "turkish_name": "kizgin"
    },
    "sad": {
      "score": 1.2,
      "turkish_name": "uzgun"
    }
    // other emotions...
  },
  "face_location": {
    "x": 120,
    "y": 85,
    "width": 200,
    "height": 200
  },
  "happiness_level": 95.2
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Testing

You can use the included example client to test the API:

```bash
python example_client.py
```

This script will:
1. Capture an image from your webcam
2. Convert it to base64
3. Send it to the API
4. Print the response

## Deployment to Railway.app

1. Install the Railway CLI and login:
```bash
npm i -g @railway/cli
railway login
```

2. Link to your Railway project:
```bash
railway link
```

3. Deploy the application:
```bash
railway up
```

## License

[MIT](LICENSE)

