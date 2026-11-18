"""Retrieval-Augmented Generation (RAG) for the farmer chat.

R – Retrieve: find the most relevant notes (TF-IDF similarity) from our trusted knowledge base
    and any extra documents in knowledge_base/docs/*.md|*.txt
A – Augment: put those notes into the prompt
G – Generate: the LLM answers using only those notes

TF-IDF is simple, fast and needs no GPU. Upgrade idea (good for your report): multilingual
sentence embeddings + a vector DB (ChromaDB) so Hindi questions match English notes better.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.client import LLMClient
from src.llm.knowledge import LANGUAGES, KnowledgeBase
from src.llm.prompts import CHAT_SYSTEM, CHAT_USER


@dataclass
class Note:
    source: str
    text: str


def _chunk(text: str, max_chars: int = 900) -> list[str]:
    chunks, current = [], ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if len(current) + len(para) > max_chars and current:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


class Retriever:
    def __init__(self, kb: KnowledgeBase, docs_dir: Path | None = None):
        self.notes = [Note(label, text) for label, text in kb.documents()]
        if docs_dir and Path(docs_dir).exists():
            for doc in sorted(Path(docs_dir).glob("*")):
                if doc.suffix.lower() in {".md", ".txt"}:
                    self.notes += [Note(doc.name, c) for c in _chunk(doc.read_text(encoding="utf-8"))]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform([n.text for n in self.notes])

    def search(self, query: str, k: int = 3, label: str | None = None) -> list[Note]:
        """Top-k notes. The note for the photo's detected disease (label) is always included first."""
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = [self.notes[i] for i in scores.argsort()[::-1] if scores[i] > 0]
        pinned = [n for n in self.notes if label and n.source == label]
        results = pinned + [n for n in ranked if n not in pinned]
        return results[:k]


class ChatAssistant:
    def __init__(self, kb: KnowledgeBase, retriever: Retriever, llm: LLMClient):
        self.kb = kb
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, lang: str = "en", label: str | None = None) -> dict[str, Any]:
        lang = lang if lang in LANGUAGES else "en"
        condition = ""
        if label and self.kb.has(label):
            info = self.kb.get(label)
            condition = f"{info['crop_en']} – {info['name_en']}"
        notes = self.retriever.search(f"{condition} {question}", k=3, label=label)
        refs = [n.source for n in notes]
        if not notes:
            fallback = ("I could not find this in my notes. Please ask your local Krishi Vigyan Kendra (KVK)."
                        if lang != "hi" else "यह जानकारी मेरे पास नहीं है। कृपया अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) से पूछें।")
            return {"answer": fallback, "source": "static", "references": []}

        user = CHAT_USER.format(
            language=LANGUAGES[lang],
            condition=condition or "none",
            notes="\n\n".join(f"[{i + 1}] {n.text}" for i, n in enumerate(notes)),
            question=question,
        )
        text = self.llm.complete(CHAT_SYSTEM, user)
        if text:
            return {"answer": text, "source": "ai", "references": refs}
        intro = ("AI chat is offline right now. Most relevant information:"
                 if lang != "hi" else "AI चैट अभी उपलब्ध नहीं है। सबसे उपयोगी जानकारी (अंग्रेज़ी में):")
        return {"answer": f"{intro}\n\n{notes[0].text}\n\n{self.kb.disclaimer(lang)}", "source": "static", "references": refs}
