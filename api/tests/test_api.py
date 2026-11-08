"""
Tes otomatis API (pytest). ID tes mengikuti tabel 5.2 template laporan (T-01, T-02, T-03).

Jalankan dari folder utama proyek:
    .venv-api\\Scripts\\python -m pytest api/tests -v
Tes memakai model asli di folder models/ (atau folder dari variabel MODEL_DIR).
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402


def make_image(fmt="JPEG", size=(256, 256), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    arr[..., 1] = np.clip(arr[..., 1] + 80, 0, 255)  # dominan hijau seperti daun
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:  # with = menjalankan lifespan (memuat model)
        yield c


def post(client, data, name="leaf.jpg", ctype="image/jpeg", **params):
    return client.post("/v1/predict", files={"file": (name, data, ctype)}, params=params)


# ----------------------------------------------------------------- health
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) == {"mobilenetv2", "efficientnetb0"}
    assert len(body["labels"]) == 3


def test_metrics_available(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "requests_total" in r.json()


# ------------------------------------------------------- T-01 input valid
def test_T01_predict_valid_jpeg_default_model(client):
    r = post(client, make_image("JPEG"))
    assert r.status_code == 200
    body = r.json()
    labels = client.get("/health").json()["labels"]
    assert body["label"] in labels
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert body["model"] == "mobilenetv2"
    assert body["inference_ms"] >= 0
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_T01_predict_valid_png_efficientnet(client):
    r = post(client, make_image("PNG"), name="leaf.png", ctype="image/png", model="efficientnetb0")
    assert r.status_code == 200
    assert r.json()["model"] == "efficientnetb0"


def test_T01_request_id_is_echoed(client):
    r = client.post(
        "/v1/predict",
        files={"file": ("leaf.jpg", make_image(), "image/jpeg")},
        headers={"X-Request-ID": "tes-abc-123"},
    )
    assert r.status_code == 200
    assert r.headers["X-Request-ID"] == "tes-abc-123"
    assert r.json()["request_id"] == "tes-abc-123"


# ----------------------------------------------------- T-02 input invalid
def test_T02_not_an_image_returns_400(client):
    r = post(client, b"ini bukan gambar", name="x.jpg", ctype="image/jpeg")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "http_400"


def test_T02_empty_file_returns_400(client):
    r = post(client, b"")
    assert r.status_code == 400


def test_T02_unsupported_content_type_returns_415(client):
    r = post(client, b"%PDF-1.4", name="x.pdf", ctype="application/pdf")
    assert r.status_code == 415


def test_T02_file_too_large_returns_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_BYTES", 1000)
    r = post(client, make_image(size=(400, 400)))
    assert r.status_code == 413


def test_T02_missing_file_returns_422(client):
    r = client.post("/v1/predict")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_T02_unknown_model_returns_400(client):
    r = post(client, make_image(), model="resnet50")
    assert r.status_code == 400


# --------------------------------------------- T-03 model tidak tersedia
def test_T03_model_unavailable_returns_503(client, monkeypatch):
    monkeypatch.setattr(main.app.state, "models", {})
    r = post(client, make_image())
    assert r.status_code == 503
    assert client.get("/health").status_code == 503
