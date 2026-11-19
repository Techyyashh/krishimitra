"""Package the app and upload it to a Hugging Face Docker Space.

Usage (PowerShell):
    $env:HF_TOKEN="hf_..."; $env:HF_SPACE_ID="your-username/krishimitra"
    python scripts/deploy_hf_space.py            # upload
    python scripts/deploy_hf_space.py --dry-run  # only build the folder and show its contents

The model itself is NOT uploaded here. In the Space settings add the variable
HF_MODEL_REPO=your-username/krishimitra-model and the server downloads it at startup.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # emoji-safe on Windows consoles
INCLUDE = ["Dockerfile", ".dockerignore", "requirements-serve.txt", "configs", "src", "api", "knowledge_base", "web", "LICENSE"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pth", "*.onnx", "*.db")


def build(target: Path) -> None:
    for name in INCLUDE:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, target / name, ignore=IGNORE)
        elif src.exists():
            shutil.copy2(src, target / name)
    (target / "models").mkdir(exist_ok=True)
    (target / "models" / ".gitkeep").touch()
    shutil.copy2(ROOT / "deploy" / "hf_space" / "README.md", target / "README.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    space_id, token = os.getenv("HF_SPACE_ID"), os.getenv("HF_TOKEN")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        build(target)
        files = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
        print(f"Prepared {len(files)} files for the Space.")
        if args.dry_run:
            print("\n".join(files))
            return
        if not space_id or not token:
            sys.exit("Set HF_SPACE_ID and HF_TOKEN environment variables first.")

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=str(target), repo_id=space_id, repo_type="space",
                          commit_message="Deploy KrishiMitra", delete_patterns=["src/**", "api/**", "web/**"])
        print(f"✅ Deployed → https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
