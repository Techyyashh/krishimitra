"""Build (and optionally deploy) the FREE static version of KrishiMitra.

The AI model runs in the visitor's browser (web/local.js + onnxruntime-web), so no server is
needed and it can be hosted free on a Hugging Face *Static* Space, GitHub Pages or Netlify.
Not included in this version (they need a server): AI chat, Claude-written advice, feedback/stats.

Usage:
    python scripts/build_static.py                                   # build into dist/static
    python scripts/build_static.py --deploy --space your-username/krishimitra
    (log in first with:  huggingface-cli login)
Test locally:
    python -m http.server 8080 -d dist/static     → http://localhost:8080
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "static"

# Windows reads text files as cp1252 by default, which breaks on emoji in the Space README.
# Re-run this script in Python's UTF-8 mode so every library reads files as UTF-8.
if os.name == "nt" and not sys.flags.utf8_mode:
    import subprocess

    sys.exit(subprocess.call([sys.executable, "-X", "utf8", __file__, *sys.argv[1:]]))

SPACE_README = """---
title: KrishiMitra
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: AI crop disease detection with advice in Hindi & English
---

# 🌾 KrishiMitra – Click a leaf, save your crop

Take a photo of a crop leaf → the AI detects the disease **right in your browser** (no upload, works offline
after the first visit) → simple advice in Hindi or English, with voice.

- Source code: [GitHub](https://github.com/{owner}/krishimitra)
- Model: [{owner}/krishimitra-model](https://huggingface.co/{owner}/krishimitra-model)

⚠️ General guidance only. Confirm with your local Krishi Vigyan Kendra (KVK) before using any chemical.
"""


def build(owner: str) -> Path:
    model, classes = ROOT / "models" / "krishimitra.onnx", ROOT / "models" / "classes.json"
    if not (model.exists() and classes.exists()):
        sys.exit("Model missing. Run scripts/import_pretrained.py or export your trained model first.")
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ROOT / "web", OUT)
    (OUT / "models").mkdir()
    shutil.copy2(model, OUT / "models" / model.name)
    shutil.copy2(classes, OUT / "models" / classes.name)
    shutil.copy2(ROOT / "knowledge_base" / "disease_info.json", OUT / "kb.json")
    (OUT / "config.js").write_text(
        '// Static build: the AI runs in the browser.\nwindow.KRISHIMITRA_API_BASE = "";\nwindow.KRISHIMITRA_MODE = "browser";\n',
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(SPACE_README.format(owner=owner), encoding="utf-8")
    files = [p for p in OUT.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"Built {len(files)} files ({size:.1f} MB) → {OUT}")
    return OUT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--space", default=os.getenv("HF_SPACE_ID"), help="e.g. your-username/krishimitra")
    args = parser.parse_args()

    owner = (args.space or "YOUR_USERNAME/x").split("/")[0]
    out = build(owner)
    if not args.deploy:
        return
    if not args.space:
        sys.exit("Pass --space your-username/krishimitra")

    from huggingface_hub import HfApi

    api = HfApi(token=os.getenv("HF_TOKEN") or None)  # None → saved `huggingface-cli login`
    api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.space, repo_type="space", commit_message="Deploy KrishiMitra (static)")
    sub = args.space.lower().replace("/", "-").replace("_", "-")
    print(f"✅ Space: https://huggingface.co/spaces/{args.space}")
    print(f"✅ App:   https://{sub}.static.hf.space")


if __name__ == "__main__":
    main()
