"""
API klasifikasi penyakit daun kentang (Early_Blight, Healthy, Late_Blight).

Model: MobileNetV2 dan EfficientNet-B0 (TensorFlow Lite float16), dilatih pada dataset PLD.

Endpoint:
    GET  /          informasi singkat layanan
    GET  /health    health check (200 jika model termuat, 503 jika tidak)
    GET  /metrics   statistik sederhana (jumlah request, error, latensi p50/p95/p99)
    POST /v1/predict  unggah satu gambar daun (field "file"), balasan berisi kelas dan keyakinan
Dokumentasi interaktif (Swagger) otomatis tersedia di /docs.

Jalankan lokal:
    python -m uvicorn main:app --host 127.0.0.1 --port 8000
"""
import io
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.exceptions import HTTPException as StarletteHTTPException

try:  # LiteRT (pengganti tflite-runtime), ringan dan tanpa TensorFlow penuh
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # pragma: no cover
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter

APP_VERSION = "1.0.0"
MODEL_VERSION = "1.0.0"
IMG_SIZE = 224
LOW_CONFIDENCE = 0.80
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_FILES = {
    "mobilenetv2": "mobilenetv2_float16.tflite",
    "efficientnetb0": "efficientnetb0_float16.tflite",
}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mobilenetv2")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"}
DEFAULT_LABELS = ["Early_Blight", "Healthy", "Late_Blight"]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("potato-api")


# ---------------------------------------------------------------- statistik
class Stats:
    """Penghitung sederhana di memori untuk monitoring (jumlah request, error, latensi)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.total = 0
        self.client_errors = 0
        self.server_errors = 0
        self.predict_latency_ms = deque(maxlen=2000)

    def add(self, path: str, status: int, ms: float):
        with self.lock:
            self.total += 1
            if 400 <= status < 500:
                self.client_errors += 1
            elif status >= 500:
                self.server_errors += 1
            if path == "/v1/predict" and status == 200:
                self.predict_latency_ms.append(ms)

    def snapshot(self):
        with self.lock:
            lat = list(self.predict_latency_ms)
            total, c4, c5 = self.total, self.client_errors, self.server_errors
        pct = {}
        if lat:
            p50, p95, p99 = np.percentile(lat, [50, 95, 99])
            pct = {"p50": round(float(p50), 2), "p95": round(float(p95), 2), "p99": round(float(p99), 2)}
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "requests_total": total,
            "client_errors_4xx": c4,
            "server_errors_5xx": c5,
            "server_error_rate": round(c5 / total, 4) if total else 0.0,
            "predict_latency_ms": pct,
            "predict_samples": len(lat),
        }


stats = Stats()


# --------------------------------------------------------------------- model
class TFLiteModel:
    """Membungkus satu interpreter TFLite. Interpreter tidak thread-safe, jadi diberi lock."""

    def __init__(self, name: str, path: Path):
        kwargs = {}
        if os.getenv("TFLITE_THREADS"):
            kwargs["num_threads"] = int(os.environ["TFLITE_THREADS"])
        self.name = name
        self.file = path.name
        self.interpreter = Interpreter(model_path=str(path), **kwargs)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.lock = threading.Lock()

    def predict(self, x: np.ndarray) -> np.ndarray:
        with self.lock:
            self.interpreter.set_tensor(self.input["index"], x.astype(self.input["dtype"]))
            self.interpreter.invoke()
            return np.array(self.interpreter.get_tensor(self.output["index"])[0], dtype=np.float64)


def load_labels() -> list[str]:
    path = MODEL_DIR / "labels.txt"
    if path.exists():
        labels = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if labels:
            return labels
    logger.warning("labels.txt tidak ditemukan, memakai label bawaan")
    return DEFAULT_LABELS


def load_models() -> dict:
    models = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        try:
            models[name] = TFLiteModel(name, path)
            logger.info("model dimuat: %s (%s)", name, path)
        except Exception as exc:  # model hilang atau rusak: layanan tetap hidup, health = 503
            logger.error("gagal memuat model %s dari %s: %s", name, path, exc)
    return models


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.labels = load_labels()
    app.state.models = load_models()
    yield


app = FastAPI(
    title="API Klasifikasi Penyakit Daun Kentang",
    version=APP_VERSION,
    description="Klasifikasi Early_Blight, Healthy, dan Late_Blight dengan MobileNetV2 dan EfficientNet-B0 (TFLite float16).",
    lifespan=lifespan,
)


# ----------------------------------------------------------- request id + log
_RID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@app.middleware("http")
async def request_context(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "")
    rid = incoming if _RID.match(incoming) else uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        ms = (time.perf_counter() - t0) * 1000
        stats.add(request.url.path, 500, ms)
        logger.exception("request_id=%s method=%s path=%s status=500 latency_ms=%.1f", rid, request.method, request.url.path, ms)
        raise
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":
        stats.add(request.url.path, response.status_code, ms)
    logger.info("request_id=%s method=%s path=%s status=%s latency_ms=%.1f", rid, request.method, request.url.path, response.status_code, ms)
    return response


def error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    rid = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": rid}},
        headers={"X-Request-ID": rid},
    )


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return error_response(request, exc.status_code, f"http_{exc.status_code}", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return error_response(request, 422, "validation_error", "Permintaan tidak valid: kirim satu gambar pada field 'file'.")


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    return error_response(request, 500, "internal_error", "Terjadi kesalahan pada server.")


# ------------------------------------------------------------ praproses citra
def preprocess(data: bytes) -> np.ndarray:
    """RGB, potong persegi di tengah, ubah ke 224x224, float32 0-255 (normalisasi ada di dalam model)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None, ...]


# ------------------------------------------------------------------ endpoints
@app.get("/", include_in_schema=False)
def root():
    return {"service": app.title, "version": APP_VERSION, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health(request: Request):
    models = request.app.state.models
    body = {
        "status": "ok" if models else "unavailable",
        "app_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "models_loaded": sorted(models),
        "labels": request.app.state.labels,
        "uptime_s": round(time.time() - stats.started, 1),
    }
    return JSONResponse(status_code=200 if models else 503, content=body)


@app.get("/metrics")
def metrics():
    return stats.snapshot()


@app.post("/v1/predict")
def predict(
    request: Request,
    file: UploadFile = File(..., description="Gambar daun kentang (JPEG, PNG, WebP, atau BMP), maksimal 5 MB."),
    model: str = Query(DEFAULT_MODEL, description="mobilenetv2 atau efficientnetb0"),
):
    if model not in MODEL_FILES:
        raise HTTPException(status_code=400, detail=f"Model tidak dikenal: {model}. Pilihan: {', '.join(MODEL_FILES)}.")
    loaded = request.app.state.models.get(model)
    if loaded is None:
        raise HTTPException(status_code=503, detail="Model belum tersedia, coba lagi nanti.")
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Tipe file tidak didukung: {file.content_type}.")

    data = file.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="File kosong.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"Ukuran file melebihi {MAX_BYTES // (1024 * 1024)} MB.")

    x = preprocess(data)
    t0 = time.perf_counter()
    probs = loaded.predict(x)
    inference_ms = (time.perf_counter() - t0) * 1000

    labels = request.app.state.labels
    best = int(np.argmax(probs))
    confidence = float(probs[best])
    return {
        "request_id": request.state.request_id,
        "model": model,
        "model_file": loaded.file,
        "model_version": MODEL_VERSION,
        "label": labels[best],
        "confidence": round(confidence, 6),
        "low_confidence": confidence < LOW_CONFIDENCE,
        "probabilities": {labels[i]: round(float(p), 6) for i, p in enumerate(probs)},
        "inference_ms": round(inference_ms, 2),
    }
