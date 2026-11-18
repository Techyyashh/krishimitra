"""Train a crop-disease classifier with two-phase transfer learning.

Phase 1 (freeze_epochs): backbone frozen, only the head learns (fast, stable).
Phase 2: everything unfrozen, small LR + cosine schedule (accuracy boost).

Usage (on Colab GPU):
    python -m src.models.train --arch efficientnet_b0
    python -m src.models.train --arch mobilenetv3_large_100
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LeafDataset, class_weights
from src.data.transforms import MEAN, STD, eval_transforms, train_transforms
from src.models.model import build_model, freeze_backbone, save_checkpoint, unfreeze_all
from src.utils.config import load_config, resolve_path, set_seed
from src.utils.logger import get_logger

log = get_logger(__name__)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None) -> tuple[float, float, float]:
    """One pass over the data. Trains if an optimizer is given. Returns (loss, accuracy, macro-F1)."""
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None and scaler.is_enabled()
    total_loss, preds, targets = 0.0, [], []
    for images, labels in tqdm(loader, leave=False, desc="train" if training else "eval"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * images.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(labels.cpu().numpy())
    y_pred, y_true = np.concatenate(preds), np.concatenate(targets)
    return (
        total_loss / len(loader.dataset),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
    )


def train(arch: str, data_root: Path, splits_dir: Path, cfg: dict) -> Path:
    tcfg = cfg["training"]
    set_seed(cfg["project"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s | arch: %s", device, arch)
    if device == "cpu":
        log.warning("No GPU found – training will be very slow. Use Google Colab (Runtime → T4 GPU).")

    classes = json.loads((splits_dir / "classes.json").read_text(encoding="utf-8"))
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    size = cfg["data"]["image_size"]

    loader_kw = dict(batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"], pin_memory=device == "cuda")
    train_loader = DataLoader(
        LeafDataset(train_df, data_root, classes, train_transforms(size)), shuffle=True, drop_last=True, **loader_kw
    )
    val_loader = DataLoader(LeafDataset(val_df, data_root, classes, eval_transforms(size)), **loader_kw)

    model = build_model(arch, len(classes)).to(device)
    weights = torch.tensor(class_weights(train_df["label"], classes), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=tcfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    out_dir = resolve_path(tcfg["output_dir"])
    ckpt_path = out_dir / f"{arch}_best.pth"
    history_path = resolve_path(tcfg["reports_dir"]) / f"{arch}_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1, patience_left = -1.0, tcfg["early_stopping_patience"]
    freeze_epochs, total_epochs = tcfg["freeze_epochs"], tcfg["epochs"]
    optimizer = scheduler = None

    with open(history_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr", "secs"])
        for epoch in range(1, total_epochs + 1):
            if epoch == 1:  # Phase 1: train only the head
                freeze_backbone(model)
                optimizer = torch.optim.AdamW(
                    (p for p in model.parameters() if p.requires_grad),
                    lr=tcfg["head_lr"], weight_decay=tcfg["weight_decay"],
                )
                scheduler = None
            if epoch == freeze_epochs + 1:  # Phase 2: fine-tune everything
                unfreeze_all(model)
                optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["finetune_lr"], weight_decay=tcfg["weight_decay"])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - freeze_epochs))
            phase = "head" if epoch <= freeze_epochs else "finetune"

            start = time.time()
            tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, device)
            lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()
            secs = time.time() - start
            writer.writerow([epoch, phase, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{va_loss:.4f}",
                             f"{va_acc:.4f}", f"{va_f1:.4f}", f"{lr:.2e}", f"{secs:.0f}"])
            fh.flush()
            log.info("Epoch %02d/%d [%s] train_acc=%.4f val_acc=%.4f val_f1=%.4f (%.0fs)",
                     epoch, total_epochs, phase, tr_acc, va_acc, va_f1, secs)

            if va_f1 > best_f1:  # keep the best model by validation macro-F1
                best_f1, patience_left = va_f1, tcfg["early_stopping_patience"]
                save_checkpoint(ckpt_path, model, {
                    "arch": arch, "classes": classes, "image_size": size,
                    "mean": list(MEAN), "std": list(STD), "val_f1": va_f1, "epoch": epoch,
                })
                log.info("  ✅ New best model saved → %s", ckpt_path)
            elif phase == "finetune":
                patience_left -= 1
                if patience_left == 0:
                    log.info("Early stopping: no improvement for %d epochs.", tcfg["early_stopping_patience"])
                    break

    log.info("Best val macro-F1: %.4f | checkpoint: %s | history: %s", best_f1, ckpt_path, history_path)
    return ckpt_path


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Train KrishiMitra disease classifier.")
    parser.add_argument("--arch", default=cfg["training"]["arch"])
    parser.add_argument("--data-root", default=cfg["data"]["plantvillage_dir"])
    parser.add_argument("--splits", default=cfg["data"]["splits_dir"])
    parser.add_argument("--epochs", type=int, help="override total epochs")
    args = parser.parse_args()
    if args.epochs:
        cfg = {**cfg, "training": {**cfg["training"], "epochs": args.epochs}}
    train(args.arch, resolve_path(args.data_root), resolve_path(args.splits), cfg)


if __name__ == "__main__":
    main()
