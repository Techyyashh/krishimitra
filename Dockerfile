# KrishiMitra API + web app. Works locally, on Hugging Face Spaces (port 7860), Render, Railway…
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# Hugging Face Spaces runs containers as user 1000 – create it so files are writable.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Install dependencies first (cached layer – rebuilds are fast when only code changes).
COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

COPY --chown=appuser:appuser . .
RUN mkdir -p data models && chown -R appuser:appuser data models

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"7860\")}/api/health', timeout=4)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
