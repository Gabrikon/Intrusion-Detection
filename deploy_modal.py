"""Modal deployment for the Acoustic Intrusion Detection app (FastAPI + custom UI).

Deploy:   modal deploy deploy_modal.py
Dev:      modal serve  deploy_modal.py
"""
import modal

APP_DIR = "/root/app"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "fastapi[standard]==0.115.6",
        "uvicorn==0.34.0",
        "python-multipart==0.0.20",
        "numpy>=2.1.0",
        "librosa>=0.11.0",
        "soundfile>=0.13.1",
        "tensorflow-cpu>=2.20.0",
    )
    .env({"MODEL_DIR": APP_DIR})
    # Bake the backend, frontend, and trained models into the image.
    .add_local_file("server.py", f"{APP_DIR}/server.py", copy=True)
    .add_local_file("inference.py", f"{APP_DIR}/inference.py", copy=True)
    .add_local_dir("static", f"{APP_DIR}/static", copy=True)
    .add_local_file("binary_model_best.keras", f"{APP_DIR}/binary_model_best.keras", copy=True)
    .add_local_file("multiclass_model_best.keras", f"{APP_DIR}/multiclass_model_best.keras", copy=True)
)

app = modal.App("acoustic-intrusion-detection", image=image)


@app.function(cpu=2.0, memory=4096, max_containers=1, scaledown_window=300)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import sys
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    from server import app as fastapi_app
    return fastapi_app
