"""Model building, freezing and checkpoint helpers (transfer learning with timm)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

SUPPORTED_ARCHS = ("efficientnet_b0", "mobilenetv3_large_100")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Load an ImageNet-pretrained backbone and replace its head with `num_classes` outputs."""
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from {SUPPORTED_ARCHS}")
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1 of transfer learning: only the new classifier head learns."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Phase 2: fine-tune the whole network with a small learning rate."""
    for p in model.parameters():
        p.requires_grad = True


def save_checkpoint(path: Path, model: nn.Module, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from a checkpoint saved by train.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
