"""Publish the exported model to the Hugging Face Hub (public model page + download source for the server).

Usage (PowerShell):
    $env:HF_TOKEN="hf_..."
    python scripts/upload_model.py --repo your-username/krishimitra-model
Uploads: models/krishimitra.onnx, models/krishimitra.int8.onnx (if present),
         models/classes.json and models/MODEL_CARD.md (as the README of the model page).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # emoji-safe on Windows consoles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="e.g. your-username/krishimitra-model")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("Set the HF_TOKEN environment variable (a Hugging Face 'write' token).")
    models = ROOT / "models"
    required = [models / "krishimitra.onnx", models / "classes.json"]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        sys.exit(f"Missing {missing}. Run: python -m src.models.export --checkpoint models/<arch>_best.pth")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    uploads = required + [models / "krishimitra.int8.onnx"]
    for path in uploads:
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.repo)
            print(f"Uploaded {path.name}")
    card = models / "MODEL_CARD.md"
    if card.exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=args.repo)
        print("Uploaded model card")
    print(f"✅ https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
