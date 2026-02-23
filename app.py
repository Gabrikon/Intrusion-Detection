import streamlit as st
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tempfile
import os
from datetime import datetime
from streamlit_mic_recorder import mic_recorder

# ============================================
# 1. CONFIGURATION & SESSION STATE
# ============================================
class Config:
    def __init__(self):
        self.sr = 22050
        self.n_mels = 128
        self.fmax = 8000
        self.duration = 1.0  # 1 second windows

conf = Config()
INTRUSION_CLASSES = ['glass_breaking', 'gun_shot', 'drilling', 'jackhammer']

if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# ============================================
# 2. DETECTOR CLASS & UTILS
# ============================================
def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=conf.sr, n_mels=conf.n_mels, fmax=conf.fmax
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram

def create_fixed_length_clips(audio, sr, num_windows=1):
    samples_per_window = int(conf.duration * sr)
    clips = []
    for i in range(num_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        if end <= len(audio):
            clips.append(audio[start:end])
        else:
            # Pad if too short
            pad = np.zeros(samples_per_window - len(audio[start:]))
            clips.append(np.concatenate([audio[start:], pad]))
    return clips

class HierarchicalIntrusionDetector:
    def __init__(self, binary_threshold=0.5):
        self.binary_model = None
        self.multiclass_model = None
        self.binary_threshold = binary_threshold
        self.intrusion_classes = INTRUSION_CLASSES

    def load_models(self, binary_path, multiclass_path):
        self.binary_model = keras.models.load_model(binary_path)
        self.multiclass_model = keras.models.load_model(multiclass_path)

    def predict(self, audio_clip, return_probs=False):
        # Create spectrogram
        spectrogram = audio_to_melspectrogram(conf, audio_clip)

        # FIX: Ensure shape is exactly (128, 48)
        if spectrogram.shape[1] > 48:
            spectrogram = spectrogram[:, :48]
        elif spectrogram.shape[1] < 48:
            pad_width = 48 - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')

        spectrogram_input = spectrogram.reshape(1, 128, 48, 1)

        # Stage 1: Binary
        binary_prob = self.binary_model.predict(spectrogram_input, verbose=0)[0][0]
        is_intrusion = binary_prob >= self.binary_threshold

        if not is_intrusion:
            if return_probs: return "normal", binary_prob, None
            return "normal"

        # Stage 2: Multiclass
        multiclass_probs = self.multiclass_model.predict(spectrogram_input, verbose=0)[0]
        intrusion_idx = np.argmax(multiclass_probs)
        if return_probs:
            return self.intrusion_classes[intrusion_idx], binary_prob, multiclass_probs
        return self.intrusion_classes[intrusion_idx]

# ============================================
# 3. STREAMLIT UI LOGIC
# ============================================
st.set_page_config(page_title="AI Intrusion Detector", page_icon="🛡️", layout="wide")

@st.cache_resource
def get_detector():
    det = HierarchicalIntrusionDetector()
    # Ensure these files exist in your directory
    det.load_models('binary_model_best.keras', 'multiclass_model_best.keras')
    return det

try:
    detector = get_detector()
except Exception as e:
    st.error(f"Error loading models: {e}. Ensure .keras files are in the folder.")
    st.stop()

st.title("🛡️ Hierarchical Intrusion Detection")
st.markdown("This system uses a two stage deep learning approach to identify security threats.")



# Sidebar
st.sidebar.header("System Controls")
detector.binary_threshold = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.5)

def process_audio(audio_data, sr):
    clips = create_fixed_length_clips(audio_data, sr, num_windows=1)
    if clips:
        result, prob, m_probs = detector.predict(clips[0], return_probs=True)

        # Update History
        timestamp = datetime.now().strftime("%H:%M:%S")
        conf_val = f"{prob:.2%}" if result != "normal" else f"{1-prob:.2%}"
        st.session_state.detection_history.insert(0, {
            "Time": timestamp,
            "Event": result.replace('_', ' ').title(),
            "Confidence": conf_val
        })

        # Display Visuals
        col1, col2 = st.columns(2)
        with col1:
            if result == "normal":
                st.success(f"### Status: SAFE \n No intrusion detected ({1-prob:.2%})")
            else:
                st.error(f"### ALERT: {result.replace('_', ' ').upper()}!")
                st.warning(f"Detection Probability: {prob:.2%}")

        with col2:
            if m_probs is not None:
                st.write("#### Threat Probability Distribution")
                chart_data = pd.DataFrame({
                    'Threat': [c.replace('_', ' ').title() for c in detector.intrusion_classes],
                    'Prob': m_probs
                })
                st.bar_chart(chart_data.set_index('Threat'))

# Tabs
tab1, tab2 = st.tabs(["📁 File Analysis", "🎤 Live Capture"])

with tab1:
    uploaded = st.file_uploader("Upload audio (WAV/MP3)", type=['wav', 'mp3'])
    if uploaded:
        st.audio(uploaded)
        y, sr = librosa.load(uploaded, sr=conf.sr)
        process_audio(y, sr)

with tab2:
    st.write("Record a sample to analyze:")
    rec = mic_recorder(start_prompt="⏺️ Record", stop_prompt="⏹️ Stop", key='rec')
    if rec:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(rec['bytes'])
            y, sr = librosa.load(tmp.name, sr=conf.sr)
        os.remove(tmp.name)
        process_audio(y, sr)

# Sidebar History Display
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Event Log")
if st.session_state.detection_history:
    st.sidebar.table(pd.DataFrame(st.session_state.detection_history))
    if st.sidebar.button("Clear Log"):
        st.session_state.detection_history = []
        st.rerun()