import streamlit as st
import requests
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
from gtts import gTTS
import base64

# ✅ UPDATE: Use your actual deployed Flask backend URL
BACKEND_URL = "https://emotion-detector-rc5h.onrender.com"
EMOTION_SERVER_URL = f"{BACKEND_URL}/predict"
COHERE_SERVER_URL = f"{BACKEND_URL}/cohere_response"

# File settings
FILENAME = "recorded_audio.wav"
DURATION = 7  # seconds
SAMPLE_RATE = 16000  # Hz

st.set_page_config(page_title="Emotion-Based AI Chatbot", layout="centered")

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .title {{
            font-size: 40px;
            font-weight: bold;
            color: white;
            text-align: center;
            padding: 10px;
        }}
        .section {{
            text-align: center;
            color: white;
            font-weight: bold;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set Background
set_background("image.png")  # Ensure image.png is in the same directory

# Page Header
st.markdown('<div class="title">🎙 Emotion-Based AI Chatbot</div>', unsafe_allow_html=True)
st.markdown("<div class='section'><h3>Speak into the microphone, and the AI will respond based on your emotion.</h3></div>", unsafe_allow_html=True)

# Function to play AI-generated response
def speak_text(text):
    try:
        tts = gTTS(text)
        tts.save("output.mp3")
        st.audio("output.mp3", format="audio/mp3")
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")

# Function to record audio
def record_audio():
    """Records audio and saves it as a WAV file."""
    st.markdown("<div class='section'>🎤 Recording...</div>", unsafe_allow_html=True)  
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    wav.write(FILENAME, SAMPLE_RATE, audio)
    st.success("✅ Recording saved!")  

# Function to convert speech to text
def convert_speech_to_text():
    recognizer = sr.Recognizer()
    with sr.AudioFile(FILENAME) as source:
        st.info("📝 Converting speech to text...")
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.success(f"🗣 Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            st.error("⚠ Could not understand the audio")
            return None
        except sr.RequestError:
            st.error("❌ Speech-to-text service unavailable")
            return None

# Function to detect emotion
def get_emotion():
    st.info("📤 Detecting emotion...")
    try:
        with open(FILENAME, "rb") as file:
            response = requests.post(EMOTION_SERVER_URL, files={"file": file})
        response_json = response.json()
        emotion = response_json.get("emotion", "unknown")
        st.success(f"🔊 Predicted Emotion: {emotion}")
        return emotion
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# Function to get AI chatbot response
def get_cohere_response(text, emotion):
    st.info("📤 Sending to AI...")
    data = {"text": text, "emotion": emotion}
    try:
        response = requests.post(COHERE_SERVER_URL, json=data)
        if response.status_code == 200:
            ai_response = response.json().get("response", "No response received.")
            st.success(f"🤖 AI: {ai_response}")
            return ai_response
        else:
            st.error(f"❌ Cohere Error: {response.json()}")
            return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# Main button to trigger recording and chatbot interaction
if st.button("🎤 Start Recording"):
    record_audio()
    text = convert_speech_to_text()
    if text:
        emotion = get_emotion()
        if emotion:
            response_text = get_cohere_response(text, emotion)
            if response_text:
                speak_text(response_text)
