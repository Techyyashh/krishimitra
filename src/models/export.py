"""Export the trained PyTorch model to ONNX (fast CPU inference, no PyTorch needed on the server),
check it matches PyTorch, and create an INT8-quantized copy (≈4× smaller) for low-end devices.

Usage:
    python -m src.models.export --checkpoint models/efficientnet_b0_best.pth
Outputs:
    models/krishimitra.onnx        (FP32 – used by the API)
    models/krishimitra.int8.onnx   (INT8 – smaller; check its accuracy before using it)
    models/classes.json            (class names + preprocessing info)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.models.model import load_checkpoint
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

log = get_logger(__name__)


def export(checkpoint: Path, onnx_path: Path, classes_path: Path) -> None:
    model, meta = load_checkpoint(checkpoint, "cpu")
    size = meta["image_size"]
    dummy = torch.randn(1, 3, size, size)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    # Verify ONNX gives the same answer as PyTorch.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    log.info("ONNX vs PyTorch max difference: %.2e (should be < 1e-3)", max_diff)

    # Benchmark CPU latency.
    for _ in range(5):
        session.run(None, {"input": dummy.numpy()})
    start = time.perf_counter()
    for _ in range(30):
        session.run(None, {"input": dummy.numpy()})
    log.info("ONNX CPU latency: %.1f ms / image", (time.perf_counter() - start) / 30 * 1000)

    int8_path = onnx_path.with_name(onnx_path.stem + ".int8.onnx")
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QUInt8)
    log.info("Sizes: FP32 %.1f MB | INT8 %.1f MB",
             onnx_path.stat().st_size / 1e6, int8_path.stat().st_size / 1e6)

    classes_path.write_text(json.dumps({
        "classes": meta["classes"], "image_size": size, "mean": meta["mean"], "std": meta["std"],
        "arch": meta["arch"], "val_f1": meta.get("val_f1"),
    }, indent=2), encoding="utf-8")
    log.info("Saved %s, %s and %s", onnx_path, int8_path, classes_path)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", default=cfg["inference"]["onnx_path"])
    parser.add_argument("--classes", default=cfg["inference"]["classes_path"])
    args = parser.parse_args()
    export(resolve_path(args.checkpoint), resolve_path(args.onnx), resolve_path(args.classes))


if __name__ == "__main__":
    main()
