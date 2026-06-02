"""FastAPI app: serves the frontend and runs intrusion-detection inference."""
import os

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import inference

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Acoustic Intrusion Detection")


@app.get("/api/health")
def health():
    return {"status": "ok", "models_available": inference.models_available()}


@app.get("/api/info")
def info():
    return {
        "classes": inference.INTRUSION_CLASSES,
        "labels": inference.CLASS_LABELS,
        "sample_rate": inference.SAMPLE_RATE,
        "window_seconds": inference.DURATION,
        "default_threshold": inference.DEFAULT_THRESHOLD,
    }


@app.post("/api/analyze-file")
async def analyze_file(file: UploadFile = File(...), threshold: float = Form(inference.DEFAULT_THRESHOLD)):
    try:
        data = await file.read()
        result = inference.analyze_file(data, file.filename or "audio", float(threshold))
        return result
    except RuntimeError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {e}"})


@app.post("/api/analyze-pcm")
async def analyze_pcm(request: Request, threshold: float = inference.DEFAULT_THRESHOLD):
    """Body = raw little-endian float32 PCM, mono, already at the model rate. Used by live + record modes."""
    try:
        raw = await request.body()
        audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if audio.size == 0:
            return JSONResponse(status_code=422, content={"error": "Empty audio buffer."})
        return inference.predict_clip(audio, float(threshold))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {e}"})


# Frontend (mounted last so /api routes take precedence).
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
