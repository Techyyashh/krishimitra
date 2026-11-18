# Contributing to KrishiMitra

Thank you for helping farmers! 🌾

## Ways to help
- 🐛 Report bugs or wrong predictions (open an issue with the photo description, never personal data)
- 🌐 Add a language: UI strings in `web/app.js` (`I18N`), knowledge-base fields in `knowledge_base/disease_info.json`
- 📚 Verify or improve disease information, **with a source** (ICAR, agri university, KVK)
- 🧠 Improve the model or add field photos (check the image license first)

## Development setup
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
ruff check .
```

## Pull requests
1. Create a branch: `git checkout -b feat/marathi-support`
2. Keep changes small and focused; add or update tests
3. `pytest` and `ruff check .` must pass (CI checks this)
4. Use clear commit messages, e.g. `feat: add Marathi UI strings`, `fix: handle EXIF rotation`

## Rules for agricultural advice
- Never add exact pesticide doses. Always point to the product label and the local KVK.
- Prefer organic and cultural measures first.
- Every new chemical recommendation must be registered for that crop in India (CIB&RC list).

By contributing you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
