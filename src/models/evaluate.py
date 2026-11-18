"""Evaluate a trained checkpoint: accuracy, macro-F1, top-3, per-class report,
confusion matrix, most-confused pairs, and CPU latency.

Usage:
    # PlantVillage test split
    python -m src.models.evaluate --checkpoint models/efficientnet_b0_best.pth
    # Real field photos (PlantDoc) – the honest robustness test
    python -m src.models.evaluate --checkpoint models/efficientnet_b0_best.pth \
        --external-dir data/raw/plantdoc/test --name plantdoc
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.dataset import LeafDataset, scan_image_folder  # noqa: E402
from src.data.labels import PLANTDOC_TO_PLANTVILLAGE  # noqa: E402
from src.data.transforms import eval_transforms  # noqa: E402
from src.models.model import load_checkpoint  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


@torch.no_grad()
def predict_all(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    probs, targets = [], []
    for images, labels in loader:
        probs.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def cpu_latency_ms(model, size: int, runs: int = 30) -> float:
    model_cpu = model.to("cpu").eval()
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        for _ in range(5):
            model_cpu(x)
        start = time.perf_counter()
        for _ in range(runs):
            model_cpu(x)
    return (time.perf_counter() - start) / runs * 1000


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path) -> None:
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    short = [c.replace("___", ": ").replace("_", " ")[:28] for c in classes]
    ax.set_xticks(range(len(classes)), short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), short, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(checkpoint: Path, df: pd.DataFrame, root: Path, name: str, reports_dir: Path) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(checkpoint, device)
    classes, size = meta["classes"], meta["image_size"]
    df = df[df["label"].isin(classes)]
    loader = DataLoader(LeafDataset(df, root, classes, eval_transforms(size)), batch_size=64, num_workers=2)

    probs, y_true = predict_all(model, loader, device)
    y_pred = probs.argmax(1)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    present = sorted(set(y_true.tolist()))  # external sets may not contain all 38 classes

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    n = len(classes)
    off_diag = [(cm[i, j], classes[i], classes[j]) for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    metrics = {
        "dataset": name,
        "arch": meta["arch"],
        "num_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted")),
        "top3_accuracy": float(np.mean([t in row for t, row in zip(y_true, top3, strict=True)])),
        "mean_confidence": float(probs.max(1).mean()),
        "cpu_latency_ms": round(cpu_latency_ms(model, size), 1),
        "most_confused": [{"count": int(c), "true": t, "predicted": p} for c, t, p in sorted(off_diag, reverse=True)[:10]],
        "per_class": classification_report(
            y_true, y_pred, labels=present, target_names=[classes[i] for i in present], output_dict=True, zero_division=0
        ),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{meta['arch']}_{name}"
    (reports_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion(cm, classes, reports_dir / f"{tag}_confusion.png")
    log.info("[%s] acc=%.4f macroF1=%.4f top3=%.4f latency=%.1fms (n=%d)", name, metrics["accuracy"],
             metrics["macro_f1"], metrics["top3_accuracy"], metrics["cpu_latency_ms"], metrics["num_images"])
    for item in metrics["most_confused"][:5]:
        log.info("  confused %3d× : %s → %s", item["count"], item["true"], item["predicted"])
    return metrics


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=cfg["data"]["plantvillage_dir"])
    parser.add_argument("--splits", default=cfg["data"]["splits_dir"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--external-dir", help="Folder of class sub-folders (e.g. PlantDoc test set)")
    parser.add_argument("--name", default=None, help="Report name (default: split name)")
    args = parser.parse_args()

    if args.external_dir:
        root = resolve_path(args.external_dir)
        df = scan_image_folder(root, label_map=PLANTDOC_TO_PLANTVILLAGE)
    else:
        root = resolve_path(args.data_root)
        df = pd.read_csv(resolve_path(args.splits) / f"{args.split}.csv")
    evaluate(resolve_path(args.checkpoint), df, root, args.name or args.split, resolve_path(cfg["training"]["reports_dir"]))


if __name__ == "__main__":
    main()
