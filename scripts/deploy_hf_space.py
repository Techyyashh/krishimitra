"""Package the app and upload it to a Hugging Face Docker Space.

Log in once (stores your token on this PC):   huggingface-cli login
Then:
    python scripts/deploy_hf_space.py --space your-username/krishimitra --model-repo your-username/krishimitra-model
    python scripts/deploy_hf_space.py --dry-run  # only build the folder and show its contents
(In GitHub Actions the HF_TOKEN and HF_SPACE_ID environment variables are used instead.)

The model itself is NOT uploaded here (see upload_model.py). --model-repo sets the Space variable
HF_MODEL_REPO, and the server downloads the model from that repo at startup.
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
    parser.add_argument("--space", default=os.getenv("HF_SPACE_ID"), help="e.g. your-username/krishimitra")
    parser.add_argument("--model-repo", help="sets the Space variable HF_MODEL_REPO")
    args = parser.parse_args()

    space_id, token = args.space, os.getenv("HF_TOKEN") or None  # None → saved `huggingface-cli login`
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        build(target)
        files = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
        print(f"Prepared {len(files)} files for the Space.")
        if args.dry_run:
            print("\n".join(files))
            return
        if not space_id:
            sys.exit("Pass --space your-username/krishimitra (or set HF_SPACE_ID).")

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
        if args.model_repo:
            api.add_space_variable(space_id, "HF_MODEL_REPO", args.model_repo)
            print(f"Set Space variable HF_MODEL_REPO={args.model_repo}")
        api.upload_folder(folder_path=str(target), repo_id=space_id, repo_type="space",
                          commit_message="Deploy KrishiMitra", delete_patterns=["src/**", "api/**", "web/**"])
        print(f"✅ Deployed → https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
