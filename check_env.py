"""Check that the development environment is installed correctly.

Run:  python check_env.py
"""
import importlib
import sys

# Windows consoles may default to cp1252, which cannot print emoji.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PACKAGES = {
    "torch": "torch", "torchvision": "torchvision", "timm": "timm", "albumentations": "albumentations",
    "opencv": "cv2", "pillow": "PIL", "numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn",
    "matplotlib": "matplotlib", "onnxruntime": "onnxruntime", "fastapi": "fastapi", "anthropic": "anthropic",
    "gtts": "gtts", "streamlit": "streamlit", "pyyaml": "yaml", "python-dotenv": "dotenv",
}

print(f"Python {sys.version.split()[0]}")
if sys.version_info[:2] != (3, 11):
    print("  ⚠️  This project is pinned for Python 3.11. Other versions may fail to install PyTorch.")

all_ok = True
for label, module_name in PACKAGES.items():
    try:
        module = importlib.import_module(module_name)
        print(f"  ✅ {label:15} {getattr(module, '__version__', '')}")
    except ImportError as err:
        all_ok = False
        print(f"  ❌ {label:15} MISSING ({err})")

if all_ok:
    import torch

    x = torch.rand(3, 4)
    print(f"\nPyTorch test: {tuple((x @ x.T).shape)} matrix computed")
    print(f"GPU available locally: {torch.cuda.is_available()} (False is fine – we train on Colab)")
    print("\n🎉 Environment ready!")
else:
    print("\n⚠️  Install the missing packages:  pip install -r requirements.txt")
