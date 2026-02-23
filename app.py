import streamlit as st
import numpy as np
import librosa
import io
import os
import time
import tempfile
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intrusion Sound Detector",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e63946, #457b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .intrusion-card {
        background: linear-gradient(135deg, #ff4d4d22, #ff000011);
        border: 2px solid #e63946;
        color: #e63946;
    }
    .normal-card {
        background: linear-gradient(135deg, #4caf5022, #2196f311);
        border: 2px solid #2a9d8f;
        color: #2a9d8f;
    }
    .confidence-bar-container {
        background: #e9ecef;
        border-radius: 999px;
        height: 10px;
        margin: 4px 0;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #457b9d, #e63946);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
    .info-box {
        background: #f0f4f8;
        border-left: 4px solid #457b9d;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .class-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-intrusion { background: #ffe0e0; color: #c0392b; }
    .badge-normal    { background: #d5f5e3; color: #1e8449; }
</style>
""", unsafe_allow_html=True)

# ─── Audio / Model Config ────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
DURATION      = 2
HOP_LENGTH    = 340 * DURATION
FMIN          = 20
FMAX          = SAMPLE_RATE // 2
N_MELS        = 128
N_FFT         = N_MELS * 20
SAMPLES       = SAMPLE_RATE * DURATION

INTRUSION_CLASSES = ['glass_breaking', 'gun_shot', 'drilling', 'jackhammer']
BINARY_THRESHOLD  = 0.5

CLASS_ICONS = {
    'glass_breaking': '🪟',
    'gun_shot':       '🔫',
    'drilling':       '🔧',
    'jackhammer':     '⚒️',
    'normal':         '✅',
}

# ─── Helper: Audio Processing ────────────────────────────────────────────────────
def audio_to_melspectrogram(audio: np.ndarray) -> np.ndarray:
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        fmin=FMIN,
        fmax=FMAX,
    )
    return librosa.power_to_db(spec)


def pad_or_crop_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Ensure spectrogram is exactly (128, 48)."""
    if spec.shape[1] > 48:
        spec = spec[:, :48]
    elif spec.shape[1] < 48:
        spec = np.pad(spec, ((0, 0), (0, 48 - spec.shape[1])), mode='constant')
    return spec


def prepare_audio_clip(audio: np.ndarray) -> np.ndarray:
    """Pad or trim audio to exactly SAMPLES length, then return mel spectrogram input."""
    if len(audio) >= SAMPLES:
        audio = audio[:SAMPLES]
    else:
        padding = SAMPLES - len(audio)
        audio = np.pad(audio, (padding // 2, padding - padding // 2), 'constant')
    spec = audio_to_melspectrogram(audio)
    spec = pad_or_crop_spectrogram(spec)
    return spec.reshape(1, 128, 48, 1)


def load_audio_from_bytes(file_bytes: bytes, filename: str = "audio") -> tuple:
    """
    Robust audio loader that tries multiple strategies:
    librosa → soundfile → raw wav fallback.
    Returns (audio_array, sample_rate) or raises RuntimeError.
    """
    errors = []

    # Strategy 1: librosa from BytesIO
    try:
        buf = io.BytesIO(file_bytes)
        audio, sr = librosa.load(buf, sr=SAMPLE_RATE, mono=True)
        if len(audio) > 0:
            return audio, sr
    except Exception as e:
        errors.append(f"librosa/BytesIO: {e}")

    # Strategy 2: soundfile from BytesIO
    try:
        buf = io.BytesIO(file_bytes)
        data, sr = sf.read(buf, always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        if len(audio) > 0:
            return audio, SAMPLE_RATE
    except Exception as e:
        errors.append(f"soundfile/BytesIO: {e}")

    # Strategy 3: write to temp file then load
    try:
        ext = os.path.splitext(filename)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        os.unlink(tmp_path)
        if len(audio) > 0:
            return audio, sr
    except Exception as e:
        errors.append(f"tempfile/librosa: {e}")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Strategy 4: soundfile on temp file
    try:
        ext = os.path.splitext(filename)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        data, sr = sf.read(tmp_path)
        os.unlink(tmp_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        if len(audio) > 0:
            return audio, SAMPLE_RATE
    except Exception as e:
        errors.append(f"tempfile/soundfile: {e}")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    raise RuntimeError(
        "Could not decode audio file. Tried:\n" + "\n".join(f"  • {e}" for e in errors)
    )


# ─── Model Loading ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(binary_path: str, multi_path: str):
    import tensorflow.keras.models as km
    binary_model = km.load_model(binary_path)
    multi_model  = km.load_model(multi_path)
    return binary_model, multi_model


# ─── Prediction ──────────────────────────────────────────────────────────────────
def predict(binary_model, multi_model, spec_input: np.ndarray) -> dict:
    binary_prob      = float(binary_model.predict(spec_input, verbose=0)[0][0])
    is_intrusion     = binary_prob >= BINARY_THRESHOLD
    multiclass_probs = None

    if is_intrusion:
        raw               = multi_model.predict(spec_input, verbose=0)[0]
        multiclass_probs  = {cls: float(raw[i]) for i, cls in enumerate(INTRUSION_CLASSES)}
        predicted_class   = max(multiclass_probs, key=multiclass_probs.get)
    else:
        predicted_class   = 'normal'

    return {
        'predicted_class':  predicted_class,
        'is_intrusion':     is_intrusion,
        'binary_prob':      binary_prob,
        'multiclass_probs': multiclass_probs,
    }


# ─── Result Display ───────────────────────────────────────────────────────────────
def show_result(result: dict):
    cls   = result['predicted_class']
    icon  = CLASS_ICONS.get(cls, '❓')
    label = cls.replace('_', ' ').title()

    if result['is_intrusion']:
        st.markdown(
            f'<div class="result-card intrusion-card">🚨 INTRUSION DETECTED<br>'
            f'<span style="font-size:1rem;font-weight:400">{icon} {label}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-card normal-card">{icon} All Clear — Normal Sound</div>',
            unsafe_allow_html=True,
        )

    st.markdown("**Detection confidence**")
    pct = int(result['binary_prob'] * 100)
    label_text = f"Intrusion probability: {pct}%"
    st.progress(result['binary_prob'], text=label_text)

    if result['multiclass_probs']:
        st.markdown("**Intrusion class breakdown**")
        for cls_name, prob in sorted(result['multiclass_probs'].items(),
                                     key=lambda x: x[1], reverse=True):
            bar_pct = int(prob * 100)
            ico     = CLASS_ICONS.get(cls_name, '•')
            st.markdown(
                f"{ico} `{cls_name.replace('_',' ').title()}` — **{bar_pct}%**"
            )
            st.progress(prob)


# ─── Sidebar: Model Setup ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Configuration")
    st.markdown("---")
    st.markdown("**Model file paths**")

    binary_path = st.text_input(
        "Binary model (.keras / .h5)",
        value="binary_model_best.keras",
        help="Path to your binary (normal vs intrusion) model file",
    )
    multi_path = st.text_input(
        "Multiclass model (.keras / .h5)",
        value="multiclass_model_best.keras",
        help="Path to your multiclass (which intrusion type) model file",
    )

    load_btn = st.button("🔄 Load / Reload Models", use_container_width=True)

    if load_btn or "models_loaded" not in st.session_state:
        if os.path.exists(binary_path) and os.path.exists(multi_path):
            with st.spinner("Loading models…"):
                try:
                    bm, mm = load_models(binary_path, multi_path)
                    st.session_state["binary_model"] = bm
                    st.session_state["multi_model"]  = mm
                    st.session_state["models_loaded"] = True
                    st.success("✅ Models loaded!")
                except Exception as e:
                    st.error(f"❌ Failed to load models:\n{e}")
                    st.session_state["models_loaded"] = False
        else:
            missing = []
            if not os.path.exists(binary_path):  missing.append(binary_path)
            if not os.path.exists(multi_path):   missing.append(multi_path)
            st.warning(f"⚠️ Model file(s) not found:\n" + "\n".join(f"• `{p}`" for p in missing))
            st.session_state["models_loaded"] = False

    st.markdown("---")
    st.markdown("**Detection threshold**")
    threshold = st.slider(
        "Intrusion sensitivity",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Lower = more sensitive (more false alarms), Higher = more strict",
    )
    BINARY_THRESHOLD = threshold

    st.markdown("---")
    st.markdown("**Intrusion classes**")
    for cls in INTRUSION_CLASSES:
        st.markdown(
            f'<span class="class-badge badge-intrusion">{CLASS_ICONS[cls]} {cls.replace("_"," ").title()}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>**Normal (background)**", unsafe_allow_html=True)
    st.markdown(
        '<span class="class-badge badge-normal">✅ Normal / Background</span>',
        unsafe_allow_html=True,
    )


# ─── Main UI ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔊 Intrusion Sound Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Hierarchical CNN — detects glass breaking, gun shots, drilling & jackhammer sounds</div>',
    unsafe_allow_html=True,
)

if not st.session_state.get("models_loaded"):
    st.info("👈 **Set your model paths in the sidebar and click Load Models** to get started.")
    st.stop()

binary_model = st.session_state["binary_model"]
multi_model  = st.session_state["multi_model"]

tab_upload, tab_record = st.tabs(["📁 Upload Audio File", "🎙️ Record Live Audio"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — FILE UPLOAD
# ════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### Upload an audio file for analysis")
    st.markdown(
        '<div class="info-box">Supported formats: WAV, MP3, OGG, FLAC, M4A, AAC. '
        'Files are processed in 2-second windows at 16 kHz.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose audio file",
        type=["wav", "mp3", "ogg", "flac", "m4a", "aac", "opus", "webm"],
        help="Max size ~200 MB",
    )

    if uploaded is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.audio(uploaded, format=uploaded.type)

        with col2:
            analyze_btn = st.button("🔍 Analyze Audio", use_container_width=True, type="primary")

        if analyze_btn:
            with st.spinner("Processing audio…"):
                try:
                    file_bytes = uploaded.read()
                    audio, sr  = load_audio_from_bytes(file_bytes, filename=uploaded.name)

                    total_duration = len(audio) / SAMPLE_RATE
                    n_windows      = max(1, int(total_duration // DURATION))
                    step           = max(1, (len(audio) - SAMPLES) // max(1, n_windows - 1))

                    all_results = []
                    for i in range(n_windows):
                        start    = i * step
                        end      = start + SAMPLES
                        clip     = audio[start:end] if end <= len(audio) else audio[start:]
                        spec_in  = prepare_audio_clip(clip)
                        res      = predict(binary_model, multi_model, spec_in)
                        res['window_start'] = start / SAMPLE_RATE
                        all_results.append(res)

                    st.success(f"✅ Analyzed {n_windows} window(s) — {total_duration:.1f}s of audio")

                    intrusions = [r for r in all_results if r['is_intrusion']]
                    if intrusions:
                        # Show worst (highest binary prob)
                        worst = max(intrusions, key=lambda r: r['binary_prob'])
                        st.markdown(f"**{len(intrusions)}/{n_windows} window(s) flagged as intrusion**")
                        show_result(worst)
                    else:
                        show_result(all_results[0])

                    if n_windows > 1:
                        with st.expander("📊 All window results"):
                            for i, r in enumerate(all_results):
                                cls  = r['predicted_class'].replace('_', ' ').title()
                                icon = CLASS_ICONS.get(r['predicted_class'], '❓')
                                t    = r['window_start']
                                pct  = int(r['binary_prob'] * 100)
                                flag = "🚨" if r['is_intrusion'] else "✅"
                                st.markdown(
                                    f"{flag} **Window {i+1}** ({t:.1f}s–{t+DURATION:.1f}s): "
                                    f"{icon} {cls} — {pct}% intrusion prob"
                                )

                except RuntimeError as e:
                    st.error(f"❌ Audio loading failed:\n\n{e}")
                    st.markdown(
                        "**Troubleshooting tips:**\n"
                        "- Make sure the file isn't corrupted\n"
                        "- Try re-exporting as WAV (PCM 16-bit, 16 kHz)\n"
                        "- Try a different browser or OS audio format"
                    )
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                    st.exception(e)

# ════════════════════════════════════════════════════════════════
# TAB 2 — LIVE RECORDING
# ════════════════════════════════════════════════════════════════
with tab_record:
    st.markdown("### Record live audio from your microphone")
    st.markdown(
        '<div class="info-box">'
        'Click <b>Start Recording</b>, make a sound, then click <b>Stop</b>. '
        'The recording will be analyzed automatically. '
        'Your browser may ask for microphone permission the first time.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Streamlit's built-in audio recorder — uses the browser's MediaRecorder API
    # Returns raw bytes (webm/opus or wav depending on browser).
    audio_bytes = st.audio_input(
        "🎙️ Click to record",
        key="live_recording",
    )

    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        st.markdown("---")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            analyze_live = st.button(
                "🔍 Analyze Recording", use_container_width=True, type="primary", key="analyze_live"
            )
        with col_b:
            st.markdown(
                '<small style="color:#6c757d;">Recording will be analyzed as a single 2-second window. '
                'For best results, keep sounds under 4 seconds.</small>',
                unsafe_allow_html=True,
            )

        if analyze_live:
            with st.spinner("Processing recording…"):
                try:
                    raw_bytes = audio_bytes.read() if hasattr(audio_bytes, 'read') else bytes(audio_bytes)

                    # Try to load with multiple fallback strategies
                    audio, sr = load_audio_from_bytes(raw_bytes, filename="recording.webm")

                    if len(audio) == 0:
                        st.error("❌ The recording appears to be empty. Please try again.")
                        st.stop()

                    duration_sec = len(audio) / SAMPLE_RATE
                    st.info(f"📏 Recording length: {duration_sec:.2f}s")

                    if duration_sec < 0.3:
                        st.warning("⚠️ Recording is very short. Please record for at least 0.5 seconds.")
                    else:
                        spec_in = prepare_audio_clip(audio)
                        result  = predict(binary_model, multi_model, spec_in)
                        show_result(result)

                except RuntimeError as e:
                    st.error(f"❌ Could not decode microphone recording.\n\n{e}")
                    st.markdown("""
**Browser audio troubleshooting:**
- Try using **Chrome** or **Edge** (best WebM/Opus support)
- Make sure you allowed microphone access
- Try a longer recording (> 1 second)
- If on Safari/Firefox, the recording format may be incompatible — try downloading and re-uploading via the **Upload** tab
- Clear browser cache and retry
                    """)
                except Exception as e:
                    st.error(f"❌ Unexpected error during analysis: {e}")
                    st.exception(e)

    else:
        st.markdown(
            '<div style="text-align:center;padding:3rem;color:#adb5bd;font-size:0.95rem;">'
            '🎙️ No recording yet — click the microphone button above to start'
            '</div>',
            unsafe_allow_html=True,
        )

# ─── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#adb5bd;font-size:0.8rem;">'
    'Hierarchical Intrusion Detection System · CNN-based · '
    'Binary + Multiclass pipeline · 16 kHz · 128-band Mel spectrograms'
    '</div>',
    unsafe_allow_html=True,
)
