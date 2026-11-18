"""Scan image folders, create stratified train/val/test splits, and serve images to PyTorch.

Run once (on Colab) to create the splits:
    python -m src.data.dataset --root data/raw/plantvillage/color
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

log = get_logger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_image_folder(root: Path, label_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns [path, label]; path is relative to root.

    label_map lets us rename folders (used to map PlantDoc names onto PlantVillage classes).
    Folders missing from label_map are skipped.
    """
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = class_dir.name
        if label_map is not None:
            if label not in label_map:
                log.warning("Skipping unmapped folder: %s", label)
                continue
            label = label_map[label]
        for img in class_dir.rglob("*"):
            if img.suffix.lower() in IMG_EXTS:
                rows.append({"path": img.relative_to(root).as_posix(), "label": label})
    if not rows:
        raise FileNotFoundError(f"No images found under {root}")
    return pd.DataFrame(rows)


def make_splits(root: Path, out_dir: Path, ratios: dict[str, float], seed: int) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split → CSV files + classes.json in out_dir."""
    df = scan_image_folder(root)
    holdout = ratios["val"] + ratios["test"]
    train_df, temp_df = train_test_split(df, test_size=holdout, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=ratios["test"] / holdout, stratify=temp_df["label"], random_state=seed
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, split in splits.items():
        split.sort_values("path").to_csv(out_dir / f"{name}.csv", index=False)
    classes = sorted(df["label"].unique())
    (out_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    log.info("Classes: %d | train %d | val %d | test %d", len(classes), *(len(s) for s in splits.values()))
    return splits


def class_weights(labels: pd.Series, classes: list[str]) -> np.ndarray:
    """Inverse-frequency weights so rare diseases matter as much as common ones."""
    counts = labels.value_counts().reindex(classes).fillna(0).to_numpy(dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(classes) * counts)
    return (weights / weights.mean()).astype(np.float32)


try:  # torch is only needed for training, not for the API server
    import torch
    from torch.utils.data import Dataset

    class LeafDataset(Dataset):
        """Reads images listed in a split DataFrame and applies Albumentations transforms."""

        def __init__(self, df: pd.DataFrame, root: Path, classes: list[str], transform=None):
            self.df = df.reset_index(drop=True)
            self.root = Path(root)
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.transform = transform

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            row = self.df.iloc[idx]
            image = np.array(Image.open(self.root / row["path"]).convert("RGB"))
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image, self.class_to_idx[row["label"]]

except ImportError:  # pragma: no cover
    LeafDataset = None  # type: ignore[assignment,misc]


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits.")
    parser.add_argument("--root", default=cfg["data"]["plantvillage_dir"], help="PlantVillage 'color' folder")
    parser.add_argument("--out", default=cfg["data"]["splits_dir"])
    args = parser.parse_args()
    make_splits(resolve_path(args.root), resolve_path(args.out), cfg["data"]["split"], cfg["project"]["seed"])


if __name__ == "__main__":
    main()
