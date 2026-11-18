"""Text-to-speech so farmers can LISTEN to advice (helps low-literacy users).

Uses gTTS (Google Translate's voice, free, needs internet). Supports Hindi and most Indian languages.
Upgrade idea: AI4Bharat Indic-TTS for offline, more natural voices.
"""
from __future__ import annotations

import hashlib
import io
import re
from collections import OrderedDict

TTS_LANGUAGES = {"en", "hi", "mr", "ta", "te", "bn", "gu", "kn", "pa", "ml", "ur"}
MAX_CHARS = 3000

_cache: OrderedDict[str, bytes] = OrderedDict()
_CACHE_SIZE = 64


class TTSError(RuntimeError):
    """Speech could not be generated (bad language, no internet, …)."""


def clean_for_speech(text: str) -> str:
    """Remove symbols that sound strange when read aloud."""
    text = re.sub(r"[*#_`>|]", " ", text)
    text = re.sub(r"^\s*-\s+", "", text, flags=re.MULTILINE)  # bullet dashes
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()[:MAX_CHARS]


def synthesize(text: str, lang: str = "en") -> bytes:
    """Return MP3 bytes for `text` spoken in `lang`."""
    if lang not in TTS_LANGUAGES:
        raise TTSError(f"Unsupported TTS language: {lang}")
    clean = clean_for_speech(text)
    if not clean:
        raise TTSError("Nothing to speak")
    key = hashlib.sha256(f"{lang}:{clean}".encode()).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    try:
        from gtts import gTTS

        buffer = io.BytesIO()
        gTTS(text=clean, lang=lang).write_to_fp(buffer)
    except Exception as err:  # gTTS raises several error types (network, HTTP, …)
        raise TTSError(f"Speech generation failed: {err}") from err
    audio = buffer.getvalue()
    _cache[key] = audio
    if len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return audio
