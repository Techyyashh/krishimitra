"""Thin wrapper around the Claude API with safe fallbacks.

If ANTHROPIC_API_KEY is missing, or the API fails / declines, `complete()` returns None
and the caller falls back to static knowledge-base text. The app never crashes because of the LLM.
"""
from __future__ import annotations

import os

from src.utils.logger import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(self, model: str, max_tokens: int = 4000, effort: str = "low"):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.enabled = bool(os.getenv("ANTHROPIC_API_KEY")) and os.getenv("LLM_ENABLED", "true").lower() == "true"
        self._client = None
        if self.enabled:
            import anthropic

            self._client = anthropic.Anthropic(max_retries=2, timeout=60.0)
        else:
            log.info("LLM disabled (no ANTHROPIC_API_KEY) – using static knowledge-base answers.")

    def complete(self, system: str, user: str) -> str | None:
        """Return the model's text answer, or None on any failure."""
        if not self.enabled or self._client is None:
            return None
        import anthropic

        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                output_config={"effort": self.effort},
                # If the model declines, the API retries on a suitable fallback model automatically.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError:
            log.warning("LLM rate limited – using static fallback")
            return None
        except anthropic.APIStatusError as err:
            log.error("LLM API error %s: %s", err.status_code, err.message)
            return None
        except anthropic.APIConnectionError:
            log.error("LLM network error – using static fallback")
            return None

        if response.stop_reason == "refusal":
            log.warning("LLM declined the request – using static fallback")
            return None
        if response.stop_reason == "max_tokens":
            log.warning("LLM answer hit max_tokens – it may be cut short")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return text or None
