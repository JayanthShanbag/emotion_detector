import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from flask import Flask, request, jsonify
import librosa
import os
import cohere

# 🔹 Initialize Cohere API
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Use Render environment variable
co = cohere.Client(COHERE_API_KEY)

# 🔹 Define Cache & Model Paths
CACHE_DIR = "./transformers_cache"
MODEL_NAME = "facebook/wav2vec2-base"
MODEL_DIR = "./models/wav2vec2-base"

# 🔹 Set Hugging Face Cache Directory
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# 🔹 Ensure model is downloaded & cached
if not os.path.exists(MODEL_DIR):
    from transformers import AutoModelForSequenceClassification, AutoProcessor
    print("Downloading model...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    model.save_pretrained(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)
    print("Model downloaded & saved.")

# 🔹 Load cached model
print("Loading cached model...")
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
print("Model loaded successfully.")

# 🔹 Initialize Flask app
app = Flask(__name__)

# 🎭 Emotion Detection Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Load audio
        y, sr = librosa.load(file, sr=16000)

        # Process input
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        
        # Get predictions
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_label = torch.argmax(logits, dim=-1).item()

        # Return response
        return jsonify({"emotion": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🤖 Cohere Chatbot Route
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Get response from Cohere API
        response = co.generate(
            model="command",  # or "command-light" for a lightweight version
            prompt=user_message,
            max_tokens=100
        )

        chatbot_reply = response.generations[0].text.strip()

        return jsonify({"reply": chatbot_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔥 Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
