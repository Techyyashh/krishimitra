"""Access to the trusted disease knowledge base (knowledge_base/disease_info.json)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.labels import PLANTVILLAGE_CLASSES, crop_key, parse_label

# Languages the app supports (code → English name used in LLM prompts).
LANGUAGES: dict[str, str] = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada", "pa": "Punjabi",
}


class KnowledgeBase:
    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta: dict[str, Any] = data["meta"]
        self.crops: dict[str, dict[str, str]] = data["crops"]
        self.healthy_template: dict[str, Any] = data["healthy_template"]
        self.diseases: dict[str, dict[str, Any]] = data["diseases"]

    def disclaimer(self, lang: str) -> str:
        return self.meta.get(f"disclaimer_{lang}", self.meta["disclaimer_en"])

    def has(self, label: str) -> bool:
        return label in self.diseases or parse_label(label)[2]

    def get(self, label: str) -> dict[str, Any]:
        """Full info card for a class label. Raises KeyError for unknown labels."""
        crop, disease, healthy = parse_label(label)
        crop_names = self.crops.get(crop_key(label), {"en": crop, "hi": crop})
        if healthy:
            entry = dict(self.healthy_template)
        elif label in self.diseases:
            entry = dict(self.diseases[label])
        else:
            raise KeyError(label)
        return {
            "label": label,
            "crop_en": crop_names["en"],
            "crop_hi": crop_names.get("hi", crop_names["en"]),
            "is_healthy": healthy,
            **entry,
        }

    def labels(self) -> list[str]:
        """All model classes that have an info card (the 38 PlantVillage classes)."""
        return [label for label in PLANTVILLAGE_CLASSES if self.has(label)]

    def documents(self) -> list[tuple[str, str]]:
        """(label, text) pairs used by the retrieval-based chat assistant."""
        docs = []
        for label in self.diseases:
            info = self.get(label)
            text = (
                f"{info['crop_en']} – {info['name_en']}. Type: {info['type']}. Cause: {info['cause']} "
                f"Symptoms: {'; '.join(info['symptoms'])}. Organic control: {'; '.join(info['organic'])}. "
                f"Chemical control: {'; '.join(info['chemical'])}. Prevention: {'; '.join(info['prevention'])}."
            )
            docs.append((label, text))
        return docs
