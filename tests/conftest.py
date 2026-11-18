"""Shared test fixtures.

Tests never need PyTorch or the real trained model: we build a tiny ONNX model
(global-average-pool → linear layer) whose output we control.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

# Must be set BEFORE the app is imported.
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["LLM_ENABLED"] = "false"

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import pytest  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.labels import PLANTVILLAGE_CLASSES  # noqa: E402

FAVOURED = "Tomato___Late_blight"


def build_dummy_model(path: Path, classes: list[str], favoured: str | None) -> None:
    """Logits = mean RGB @ W + b. A big bias on `favoured` → confident prediction; none → uniform."""
    n = len(classes)
    weights = np.zeros((3, n), dtype=np.float32)
    bias = np.zeros(n, dtype=np.float32)
    if favoured:
        bias[classes.index(favoured)] = 10.0
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "W", "B"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, "dummy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, "h", "w"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", n])],
        initializer=[numpy_helper.from_array(weights, "W"), numpy_helper.from_array(bias, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def write_model(tmp: Path, favoured: str | None) -> tuple[Path, Path]:
    onnx_path, classes_path = tmp / "model.onnx", tmp / "classes.json"
    build_dummy_model(onnx_path, PLANTVILLAGE_CLASSES, favoured)
    classes_path.write_text(json.dumps({"classes": PLANTVILLAGE_CLASSES, "image_size": 64}), encoding="utf-8")
    return onnx_path, classes_path


@pytest.fixture
def confident_model(tmp_path: Path) -> tuple[Path, Path]:
    return write_model(tmp_path, FAVOURED)


@pytest.fixture
def uncertain_model(tmp_path: Path) -> tuple[Path, Path]:
    return write_model(tmp_path, None)


def solid_image(color: tuple[int, int, int], size: int = 128) -> Image.Image:
    return Image.new("RGB", (size, size), color)


def leaf_like_image() -> Image.Image:
    """Green 'leaf' with brown spots on a grey background."""
    img = solid_image((120, 120, 120), 200)
    rng = np.random.default_rng(0)
    arr = np.array(img)
    arr[30:170, 40:160] = (40, 140, 50)
    for _ in range(15):
        y, x = rng.integers(40, 160, size=2)
        arr[y : y + 6, x : x + 6] = (110, 70, 30)
    return Image.fromarray(arr)


def to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
