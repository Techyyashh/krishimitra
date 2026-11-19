"""Production inference with ONNX Runtime (no PyTorch needed → small, fast server).

Preprocessing here MUST match eval_transforms() used in training: resize → /255 → ImageNet normalize.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.labels import parse_label
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PredictionItem:
    label: str
    crop: str
    disease: str
    is_healthy: bool
    confidence: float


class Predictor:
    """Loads the ONNX model once and classifies PIL images."""

    def __init__(self, onnx_path: Path, classes_path: Path, top_k: int = 3, threshold: float = 0.6):
        import onnxruntime as ort

        meta = json.loads(Path(classes_path).read_text(encoding="utf-8"))
        self.classes: list[str] = meta["classes"]
        self.size: int = int(meta.get("image_size", 224))
        # Optional: resize the shortest edge first, then centre-crop to `size`
        # (used by models trained with that preprocessing, e.g. Hugging Face MobileNetV2).
        self.resize_shortest: int | None = meta.get("resize_shortest_edge")
        self.mean = np.array(meta.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(meta.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.top_k = min(top_k, len(self.classes))
        self.threshold = threshold

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        log.info("Loaded model %s (%d classes, %dpx)", Path(onnx_path).name, len(self.classes), self.size)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        if self.resize_shortest:
            w, h = rgb.size
            scale = self.resize_shortest / min(w, h)
            rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
            w, h = rgb.size
            left, top = (w - self.size) // 2, (h - self.size) // 2
            rgb = rgb.crop((left, top, left + self.size, top + self.size))
        else:
            rgb = rgb.resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(rgb, dtype=np.float32) / 255.0
        x = (x - self.mean) / self.std
        return np.transpose(x, (2, 0, 1))[np.newaxis].astype(np.float32)  # HWC → 1×C×H×W

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max()
        e = np.exp(z)
        return e / e.sum()

    def predict(self, image: Image.Image) -> dict[str, Any]:
        start = time.perf_counter()
        logits = self.session.run(None, {self.input_name: self.preprocess(image)})[0][0]
        probs = self._softmax(logits.astype(np.float64))
        order = np.argsort(probs)[::-1][: self.top_k]
        top = []
        for idx in order:
            label = self.classes[int(idx)]
            crop, disease, healthy = parse_label(label)
            top.append(asdict(PredictionItem(label, crop, disease, healthy, round(float(probs[idx]), 4))))
        return {
            "top": top,
            "best": top[0],
            "confident": top[0]["confidence"] >= self.threshold,
            "inference_ms": round((time.perf_counter() - start) * 1000, 1),
        }


def download_model_if_missing(onnx_path: Path, classes_path: Path) -> None:
    """On a fresh server (e.g. Hugging Face Space) pull the model from the HF Hub.

    Set HF_MODEL_REPO=your-username/krishimitra-model in the hosting platform's variables.
    """
    repo = os.getenv("HF_MODEL_REPO")
    if (onnx_path.exists() and classes_path.exists()) or not repo:
        return
    from huggingface_hub import hf_hub_download

    for target in (onnx_path, classes_path):
        if not target.exists():
            log.info("Downloading %s from %s", target.name, repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id=repo, filename=target.name, local_dir=str(target.parent),
                            token=os.getenv("HF_TOKEN") or None)


def load_predictor(onnx_path: Path, classes_path: Path, top_k: int, threshold: float) -> Predictor | None:
    """Return a Predictor, or None (with a clear log message) if the model is not available yet."""
    try:
        download_model_if_missing(onnx_path, classes_path)
    except Exception as err:  # network / auth problems must not crash the server
        log.error("Model download failed: %s", err)
    if not (onnx_path.exists() and classes_path.exists()):
        log.warning("Model not found at %s. Train + export it first (see README → Train the model).", onnx_path)
        return None
    return Predictor(onnx_path, classes_path, top_k, threshold)
