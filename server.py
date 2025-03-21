import os
import torch
import librosa
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Model and API settings
MODEL_NAME = "facebook/wav2vec2-base"
CACHE_DIR = "./transformers_cache"

# Set cache directory to avoid excessive downloads
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# Load Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model.eval()

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize Cohere API
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(cohere_api_key) if cohere_api_key else None

@app.route("/predict", methods=["POST"])
def predict():
    """Handles emotion detection from audio"""
    try:
        file = request.files["file"]
        audio, sr = librosa.load(file, sr=16000)
        input_values = processor(audio, sampling_rate=sr, return_tensors="pt").input_values
        input_values = input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits
            predicted_label = torch.argmax(logits, dim=-1).item()

        emotions = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]
        emotion = emotions[predicted_label]

        return jsonify({"emotion": emotion})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chatbot responses with Cohere"""
    if not cohere_client:
        return jsonify({"error": "Cohere API key is missing"})

    try:
        data = request.json
        user_input = data.get("message", "")

        response = cohere_client.chat(user_input)
        return jsonify({"response": response.text})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
