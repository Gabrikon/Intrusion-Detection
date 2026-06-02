"""Audio preprocessing + two-stage CNN inference for the intrusion detector.

Stage 1 (binary): normal vs. intrusion.
Stage 2 (multiclass): which intrusion type, only run when stage 1 fires.
"""
import io
import os
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ─── Audio / model config ──────────────────────────────────────────────────
# These MUST match the values the models were trained with (see app.ipynb):
# sr=22050, 1-second windows, librosa default hop_length/n_fft/fmin,
# n_mels=128, fmax=8000, and power_to_db(ref=np.max), padded to 48 frames.
SAMPLE_RATE = 22050
DURATION = 1
FMAX = 8000
N_MELS = 128
SAMPLES = SAMPLE_RATE * DURATION
SPEC_FRAMES = 48

INTRUSION_CLASSES = ["glass_breaking", "gun_shot", "drilling", "jackhammer"]
DEFAULT_THRESHOLD = 0.5

CLASS_LABELS = {
    "glass_breaking": "Glass Breaking",
    "gun_shot": "Gun Shot",
    "drilling": "Drilling",
    "jackhammer": "Jackhammer",
    "normal": "Normal / Background",
}

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
BINARY_MODEL_PATH = os.path.join(MODEL_DIR, "binary_model_best.keras")
MULTI_MODEL_PATH = os.path.join(MODEL_DIR, "multiclass_model_best.keras")

_models = {"binary": None, "multi": None}


def models_available() -> bool:
    return os.path.exists(BINARY_MODEL_PATH) and os.path.exists(MULTI_MODEL_PATH)


def load_models():
    """Lazily load and cache both Keras models."""
    if _models["binary"] is None or _models["multi"] is None:
        import tensorflow.keras.models as km  # imported lazily (TF is heavy)
        _models["binary"] = km.load_model(BINARY_MODEL_PATH)
        _models["multi"] = km.load_model(MULTI_MODEL_PATH)
    return _models["binary"], _models["multi"]


# ─── Feature extraction ─────────────────────────────────────────────────────
def _melspectrogram(audio: np.ndarray) -> np.ndarray:
    import librosa
    # Match training exactly: librosa defaults for hop_length/n_fft/fmin,
    # and power_to_db normalized to the per-clip maximum (ref=np.max).
    spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=FMAX,
    )
    return librosa.power_to_db(spec, ref=np.max)


def _fit_frames(spec: np.ndarray) -> np.ndarray:
    if spec.shape[1] > SPEC_FRAMES:
        return spec[:, :SPEC_FRAMES]
    if spec.shape[1] < SPEC_FRAMES:
        return np.pad(spec, ((0, 0), (0, SPEC_FRAMES - spec.shape[1])), mode="constant")
    return spec


def prepare_clip(audio: np.ndarray) -> np.ndarray:
    """Pad/trim a mono 16 kHz waveform to one 2s window → model input tensor."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) >= SAMPLES:
        audio = audio[:SAMPLES]
    else:
        pad = SAMPLES - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2), "constant")
    spec = _fit_frames(_melspectrogram(audio))
    return spec.reshape(1, N_MELS, SPEC_FRAMES, 1)


# ─── Prediction ───────────────────────────────────────────────────────────
def predict_clip(audio: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    binary_model, multi_model = load_models()
    spec = prepare_clip(audio)

    binary_prob = float(binary_model.predict(spec, verbose=0)[0][0])
    is_intrusion = binary_prob >= threshold
    multiclass_probs = None
    predicted_class = "normal"

    if is_intrusion:
        raw = multi_model.predict(spec, verbose=0)[0]
        multiclass_probs = {cls: float(raw[i]) for i, cls in enumerate(INTRUSION_CLASSES)}
        predicted_class = max(multiclass_probs, key=multiclass_probs.get)

    return {
        "is_intrusion": is_intrusion,
        "predicted_class": predicted_class,
        "label": CLASS_LABELS.get(predicted_class, predicted_class),
        "binary_prob": binary_prob,
        "multiclass_probs": multiclass_probs,
    }


# ─── Robust file decoding ───────────────────────────────────────────────────
def decode_audio(file_bytes: bytes, filename: str = "audio") -> np.ndarray:
    """Decode arbitrary audio bytes → mono float32 at 16 kHz. Raises RuntimeError."""
    import librosa
    import soundfile as sf

    errors = []
    tmp_path = None

    try:
        audio, _ = librosa.load(io.BytesIO(file_bytes), sr=SAMPLE_RATE, mono=True)
        if len(audio) > 0:
            return audio
    except Exception as e:
        errors.append(f"librosa/BytesIO: {e}")

    try:
        data, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        if len(audio) > 0:
            return audio
    except Exception as e:
        errors.append(f"soundfile/BytesIO: {e}")

    for loader in ("librosa", "soundfile"):
        try:
            ext = os.path.splitext(filename)[-1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            if loader == "librosa":
                audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
            else:
                data, sr = sf.read(tmp_path)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                audio = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(audio) > 0:
                return audio
        except Exception as e:
            errors.append(f"tempfile/{loader}: {e}")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                tmp_path = None

    raise RuntimeError("Could not decode audio. Tried:\n" + "\n".join(f"  - {e}" for e in errors))


def analyze_file(file_bytes: bytes, filename: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Decode a full file, scan it in overlapping 2s windows, return aggregate result."""
    audio = decode_audio(file_bytes, filename)
    total_duration = len(audio) / SAMPLE_RATE
    n_windows = max(1, int(total_duration // DURATION))
    step = max(1, (len(audio) - SAMPLES) // max(1, n_windows - 1)) if n_windows > 1 else 0

    windows = []
    for i in range(n_windows):
        start = i * step
        end = start + SAMPLES
        clip = audio[start:end] if end <= len(audio) else audio[start:]
        res = predict_clip(clip, threshold)
        res["window_start"] = round(start / SAMPLE_RATE, 2)
        windows.append(res)

    intrusions = [w for w in windows if w["is_intrusion"]]
    summary = max(intrusions, key=lambda w: w["binary_prob"]) if intrusions else windows[0]

    return {
        **{k: summary[k] for k in ("is_intrusion", "predicted_class", "label", "binary_prob", "multiclass_probs")},
        "duration": round(total_duration, 2),
        "n_windows": n_windows,
        "n_flagged": len(intrusions),
        "windows": windows,
    }
