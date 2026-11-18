"""API tests using FastAPI's TestClient with a dummy ONNX model and a temporary database."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FAVOURED, leaf_like_image, solid_image, to_jpeg_bytes


@pytest.fixture
def client(confident_model, tmp_path, monkeypatch):
    onnx_path, classes_path = confident_model
    monkeypatch.setenv("KRISHIMITRA_ONNX_PATH", str(onnx_path))
    monkeypatch.setenv("KRISHIMITRA_CLASSES_PATH", str(classes_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def upload(client, data: bytes, content_type: str = "image/jpeg"):
    return client.post("/api/predict", files={"file": ("leaf.jpg", data, content_type)})


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["llm_enabled"] is False


def test_predict_leaf(client):
    res = upload(client, to_jpeg_bytes(leaf_like_image()))
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["prediction"]["label"] == FAVOURED
    assert body["prediction"]["disease_hi"]
    assert body["info"]["severity"] == "high"
    assert body["scan_id"] >= 1


def test_predict_not_leaf(client):
    body = upload(client, to_jpeg_bytes(solid_image((20, 40, 220)))).json()
    assert body["status"] == "not_leaf"
    assert body["prediction"] is None


def test_predict_rejects_wrong_type(client):
    assert upload(client, b"hello", "text/plain").status_code == 415


def test_predict_rejects_corrupt_image(client):
    assert upload(client, b"not really a jpeg").status_code == 400


def test_predict_rejects_huge_file(client):
    assert upload(client, b"0" * (9 * 1024 * 1024)).status_code == 413


def test_advice_static(client):
    res = client.post("/api/advice", json={"label": FAVOURED, "lang": "hi"})
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "static"
    assert "KVK" in body["disclaimer"]


def test_advice_validation(client):
    assert client.post("/api/advice", json={"label": FAVOURED, "lang": "xx"}).status_code == 422
    assert client.post("/api/advice", json={"label": "Banana___Panama"}).status_code == 404


def test_chat(client):
    res = client.post("/api/chat", json={"question": "Can I use neem oil?", "lang": "en", "label": FAVOURED})
    assert res.status_code == 200
    assert res.json()["references"][0] == FAVOURED


def test_feedback_and_stats(client):
    scan_id = upload(client, to_jpeg_bytes(leaf_like_image())).json()["scan_id"]
    res = client.post("/api/feedback", json={"scan_id": scan_id, "predicted_label": FAVOURED, "correct": False})
    assert res.status_code == 200
    stats = client.get("/api/stats").json()
    assert stats["total_scans"] >= 1
    assert stats["feedback_count"] == 1
    assert stats["most_reported_wrong"][0]["label"] == FAVOURED


def test_labels(client):
    labels = client.get("/api/labels").json()
    assert len(labels) == 38


def test_web_app_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "KrishiMitra" in res.text
