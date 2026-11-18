"""Grad-CAM: show WHICH part of the leaf the model looked at.

Usage:
    python -m src.models.gradcam --checkpoint models/efficientnet_b0_best.pth \
        --image path/to/leaf.jpg --out reports/gradcam_example.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.transforms import eval_transforms
from src.models.model import load_checkpoint
from src.utils.config import resolve_path


def gradcam_overlay(model, meta: dict, image: Image.Image) -> tuple[np.ndarray, str, float]:
    """Return (overlay RGB uint8 image, predicted label, confidence)."""
    size = meta["image_size"]
    rgb = np.array(image.convert("RGB").resize((size, size)))
    tensor = eval_transforms(size)(image=rgb)["image"].unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(probs.argmax())
    # The last block of EfficientNet / MobileNetV3 holds the richest spatial features.
    with GradCAM(model=model, target_layers=[model.blocks[-1]]) as cam:
        heatmap = cam(input_tensor=tensor)[0]
    overlay = show_cam_on_image(rgb.astype(np.float32) / 255.0, heatmap, use_rgb=True)
    return overlay, meta["classes"][idx], float(probs[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Grad-CAM heatmap for one leaf image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="reports/gradcam_example.jpg")
    args = parser.parse_args()

    model, meta = load_checkpoint(resolve_path(args.checkpoint), "cpu")
    overlay, label, conf = gradcam_overlay(model, meta, Image.open(args.image))
    out = resolve_path(args.out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out)
    print(f"Prediction: {label} ({conf:.1%}) → heatmap saved to {out}")


if __name__ == "__main__":
    main()
