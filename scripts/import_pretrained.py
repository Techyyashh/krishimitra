"""Import an open, already-trained PlantVillage model as a STARTER model, so the app works
before you train your own on Colab.

Default source: neghani/plantvillage-mobilenetv2 (Apache-2.0, MobileNetV2, same 38 class names).
Its model card reports 99.2% on the PlantVillage test split but only ~16% on 25 real field
photos – replace it with your own trained model (notebooks/train_on_colab.ipynb) when you can.

Usage:
    pip install transformers==4.46.3
    python scripts/import_pretrained.py
Writes models/krishimitra.onnx and models/classes.json (the files the app loads).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.labels import PLANTVILLAGE_CLASSES  # noqa: E402
from src.inference.predictor import Predictor  # noqa: E402


class LogitsOnly(torch.nn.Module):
    """Hugging Face models return an object; ONNX export needs a plain tensor."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="neghani/plantvillage-mobilenetv2")
    parser.add_argument("--out-dir", default=str(ROOT / "models"))
    args = parser.parse_args()

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(args.repo).eval()
    processor = AutoImageProcessor.from_pretrained(args.repo)
    classes = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if sorted(classes) != sorted(PLANTVILLAGE_CLASSES):
        sys.exit("This model's classes do not match the 38 PlantVillage classes KrishiMitra knows.")

    size = processor.crop_size["height"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path, classes_path = out / "krishimitra.onnx", out / "classes.json"

    # .eval() matters: export restores the wrapper's mode afterwards, and a new module starts in training mode.
    wrapped = LogitsOnly(model).eval()
    torch.onnx.export(
        wrapped, torch.randn(1, 3, size, size), str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
    )
    classes_path.write_text(json.dumps({
        "classes": classes,
        "image_size": size,
        "resize_shortest_edge": processor.size["shortest_edge"],
        "mean": list(processor.image_mean),
        "std": list(processor.image_std),
        "arch": "mobilenet_v2_1.0_224",
        "source": f"https://huggingface.co/{args.repo}",
        "license": "apache-2.0",
        "note": "Starter model. Replace with your own trained model.",
    }, indent=2), encoding="utf-8")

    # Check 1: ONNX gives the same logits as PyTorch.
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        ref = wrapped.eval()(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    diff = float(np.abs(sess.run(None, {"input": x.numpy()})[0] - ref).max())
    print(f"ONNX vs PyTorch max difference: {diff:.2e}")

    # Check 2: our Predictor's preprocessing matches the model's own image processor.
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (300, 420, 3), dtype=np.uint8))
    hf_input = processor(images=img, return_tensors="np")["pixel_values"]
    ours = Predictor(onnx_path, classes_path).preprocess(img)
    print(f"Preprocessing difference vs Hugging Face processor (mean abs): {np.abs(hf_input - ours).mean():.4f}")
    print(f"✅ Saved {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB) and {classes_path.name} to {out}")


if __name__ == "__main__":
    main()
