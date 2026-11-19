"""Publish the exported model to the Hugging Face Hub (public model page + download source for the server).

Log in once (stores your token on this PC):   huggingface-cli login
Then:
    python scripts/upload_model.py --repo your-username/krishimitra-model
(Alternatively set the HF_TOKEN environment variable instead of logging in.)

Uploads: models/krishimitra.onnx, models/krishimitra.int8.onnx (if present), models/classes.json
and a model card as the page's README. If classes.json says the model is the imported STARTER
model, a card crediting the original authors is generated instead of models/MODEL_CARD.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # emoji-safe on Windows consoles

STARTER_CARD = """---
license: apache-2.0
library_name: onnx
pipeline_tag: image-classification
tags: [agriculture, plant-disease, mobilenetv2, onnx, plantvillage]
base_model: {source_id}
---

# KrishiMitra – starter model (ONNX)

ONNX conversion of [{source_id}]({source}) (Apache-2.0) used by the
[KrishiMitra app](https://github.com/{owner}/krishimitra) until its own model is trained.
All credit for training goes to the original authors – see their model card for data, metrics and limitations.

- Architecture: MobileNetV2 (1.0, 224), 38 PlantVillage classes (`classes.json`)
- Preprocessing: resize shortest edge to {resize}, centre-crop {size}, normalise mean {mean} / std {std}
- Reported by the original card: 99.2% accuracy on the PlantVillage test split, **~16% on a small real-field set**

⚠️ Decision support only. Confirm with a local agricultural expert (KVK) before using any chemical.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="e.g. your-username/krishimitra-model")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    models = ROOT / "models"
    required = [models / "krishimitra.onnx", models / "classes.json"]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        sys.exit(f"Missing {missing}. Run scripts/import_pretrained.py or export your trained checkpoint first.")

    from huggingface_hub import HfApi

    api = HfApi(token=os.getenv("HF_TOKEN") or None)  # None → use the login saved by `huggingface-cli login`
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for path in required + [models / "krishimitra.int8.onnx"]:
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.repo)
            print(f"Uploaded {path.name}")

    meta = json.loads((models / "classes.json").read_text(encoding="utf-8"))
    if meta.get("source"):
        source_id = meta["source"].removeprefix("https://huggingface.co/")
        card = STARTER_CARD.format(source=meta["source"], source_id=source_id, owner=args.repo.split("/")[0],
                                   resize=meta.get("resize_shortest_edge"), size=meta["image_size"],
                                   mean=meta["mean"], std=meta["std"])
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=args.repo)
        print("Uploaded starter model card (credits the original authors)")
    elif (models / "MODEL_CARD.md").exists():
        api.upload_file(path_or_fileobj=str(models / "MODEL_CARD.md"), path_in_repo="README.md", repo_id=args.repo)
        print("Uploaded model card")
    print(f"✅ https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
