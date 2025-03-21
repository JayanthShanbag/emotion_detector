import os
import torch
import cohere
import librosa
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from dotenv import load_dotenv

# Load environment variables (Cohere API key)
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere API
co = cohere.Client(COHERE_API_KEY)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Model setup
MODEL_NAME = "facebook/wav2vec2-base"
CACHE_DIR = "./transformers_cache"

# Load processor & model
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=7, cache_dir=CACHE_DIR)
model.eval()

# Emotion Labels
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

@app.route("/")
def home():
    return jsonify({"status": "Server is running!"})

@app.route("/chat", methods=["POST"])
def chat():
    """Chatbot endpoint using Cohere."""
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        # FIX: Corrected Cohere API call
        response = co.chat(user_message)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    """Predicts emotion from an uploaded audio file."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    audio_path = os.path.join("/tmp", audio_file.filename)
    audio_file.save(audio_path)

    try:
        # Load and preprocess audio
        speech, sr = librosa.load(audio_path, sr=16000)
        input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

        # Predict emotion
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_class = torch.argmax(logits, dim=1).item()

        predicted_emotion = EMOTIONS[predicted_class]
        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
