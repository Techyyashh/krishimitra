"""Per-IP rate limiting (protects the free server and your LLM bill from abuse)."""
from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.utils.config import load_config

_cfg = load_config()["api"]
PREDICT_LIMIT: str = _cfg["rate_limit"]
LLM_LIMIT: str = _cfg["llm_rate_limit"]

limiter = Limiter(
    key_func=get_remote_address,
    enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
)
