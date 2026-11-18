"""Turns a prediction into farmer-friendly advice, in the farmer's language.

Order of preference:
1. Cached answer (same disease + language asked before → free and instant)
2. LLM answer grounded in the knowledge base
3. Static knowledge-base text (always works, even offline / without API key)
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from src.llm.client import LLMClient
from src.llm.knowledge import LANGUAGES, KnowledgeBase
from src.llm.prompts import ADVICE_SYSTEM, ADVICE_USER

_HEADINGS = {
    "en": ("What is it?", "Why did it happen?", "What to do now", "How to prevent it next time",
           "Chemical options (only after checking with KVK)"),
    "hi": ("यह क्या है?", "यह क्यों हुआ?", "अभी क्या करें", "अगली बार कैसे बचें",
           "रासायनिक विकल्प (KVK से पुष्टि के बाद ही)"),
}


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


class Advisor:
    def __init__(self, kb: KnowledgeBase, llm: LLMClient, cache_size: int = 256):
        self.kb = kb
        self.llm = llm
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._cache_size = cache_size

    def static_text(self, label: str, lang: str = "en") -> str:
        """Readable advice built only from the knowledge base (no AI)."""
        info = self.kb.get(label)
        h = _HEADINGS.get(lang, _HEADINGS["en"])
        parts = []
        if lang == "hi":
            parts.append(f"{info['crop_hi']} – {info['name_hi']}")
            parts.append(info["summary_hi"])
            parts.append("विस्तृत जानकारी (अंग्रेज़ी में):")
        else:
            parts.append(f"{info['crop_en']} – {info['name_en']}")
        parts += [
            f"{h[0]}\n{_bullets(info['symptoms'])}",
            f"{h[1]}\n- {info['cause']}",
            f"{h[2]}\n{_bullets(info['organic'])}",
            f"{h[4]}\n{_bullets(info['chemical'])}",
            f"{h[3]}\n{_bullets(info['prevention'])}",
            self.kb.disclaimer(lang),
        ]
        return "\n\n".join(parts)

    def advise(self, label: str, lang: str = "en", confidence: float = 1.0) -> dict[str, Any]:
        """Return {'text', 'source': 'ai'|'static'|'cache', 'disclaimer'}. Raises KeyError for unknown labels."""
        info = self.kb.get(label)  # validates the label
        lang = lang if lang in LANGUAGES else "en"
        disclaimer = self.kb.disclaimer(lang)
        # Low-confidence answers are not cached, because their wording differs.
        cache_key = (label, lang)
        if confidence >= 0.7 and cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return {"text": self._cache[cache_key], "source": "cache", "disclaimer": disclaimer}

        reference = {k: info[k] for k in ("crop_en", "name_en", "type", "cause", "symptoms",
                                          "organic", "chemical", "prevention", "severity")}
        user = ADVICE_USER.format(
            language=LANGUAGES[lang],
            crop=info["crop_en"],
            condition=info["name_en"],
            confidence=round(confidence * 100),
            reference=json.dumps(reference, ensure_ascii=False, indent=1),
        )
        text = self.llm.complete(ADVICE_SYSTEM, user)
        if text:
            if confidence >= 0.7:
                self._cache[cache_key] = text
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return {"text": text, "source": "ai", "disclaimer": disclaimer}
        return {"text": self.static_text(label, lang), "source": "static", "disclaimer": disclaimer}
