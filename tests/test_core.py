"""Unit tests for labels, leaf validator, predictor, knowledge base, advisor, RAG and TTS cleaning."""
from __future__ import annotations

from src.data.labels import PLANTDOC_TO_PLANTVILLAGE, PLANTVILLAGE_CLASSES, parse_label
from src.inference.leaf_validator import is_probably_leaf
from src.inference.predictor import Predictor
from src.llm.advisor import Advisor
from src.llm.client import LLMClient
from src.llm.knowledge import KnowledgeBase
from src.llm.rag import ChatAssistant, Retriever
from src.speech.tts import clean_for_speech
from src.utils.config import load_config, resolve_path
from tests.conftest import FAVOURED, leaf_like_image, solid_image

KB = KnowledgeBase(resolve_path(load_config()["knowledge_base"]["path"]))


def offline_llm() -> LLMClient:
    llm = LLMClient("claude-opus-5-5")
    assert not llm.enabled  # conftest sets LLM_ENABLED=false
    return llm


# ---------- labels ----------
def test_parse_label_variants():
    assert parse_label("Pepper,_bell___Bacterial_spot") == ("Pepper, bell", "Bacterial spot", False)
    assert parse_label("Corn_(maize)___Common_rust_") == ("Corn (maize)", "Common rust", False)
    assert parse_label("Apple___healthy") == ("Apple", "Healthy", True)


def test_class_list_and_mapping_consistent():
    assert len(PLANTVILLAGE_CLASSES) == 38
    assert PLANTVILLAGE_CLASSES == sorted(PLANTVILLAGE_CLASSES)
    assert set(PLANTDOC_TO_PLANTVILLAGE.values()) <= set(PLANTVILLAGE_CLASSES)


# ---------- leaf validator ----------
def test_leaf_validator_accepts_leaf_and_rejects_others():
    assert is_probably_leaf(leaf_like_image())[0]
    assert not is_probably_leaf(solid_image((20, 40, 220)))[0]   # blue
    assert not is_probably_leaf(solid_image((128, 128, 128)))[0]  # grey


# ---------- predictor ----------
def test_predictor_confident(confident_model):
    predictor = Predictor(*confident_model, top_k=3, threshold=0.6)
    result = predictor.predict(leaf_like_image())
    assert result["best"]["label"] == FAVOURED
    assert result["confident"] is True
    assert len(result["top"]) == 3
    confs = [item["confidence"] for item in result["top"]]
    assert confs == sorted(confs, reverse=True)


def test_predictor_uncertain(uncertain_model):
    predictor = Predictor(*uncertain_model, top_k=3, threshold=0.6)
    result = predictor.predict(leaf_like_image())
    assert result["confident"] is False
    assert abs(result["best"]["confidence"] - 1 / 38) < 0.01


# ---------- knowledge base ----------
def test_knowledge_base_covers_all_classes():
    required = {"name_en", "name_hi", "cause", "symptoms", "organic", "chemical", "prevention", "severity", "summary_hi"}
    for label in PLANTVILLAGE_CLASSES:
        info = KB.get(label)
        assert required <= info.keys(), label
        assert info["crop_hi"], label


# ---------- advisor ----------
def test_advisor_static_fallback_english_and_hindi():
    advisor = Advisor(KB, offline_llm())
    en = advisor.advise("Potato___Late_blight", "en", 0.95)
    assert en["source"] == "static"
    assert "Potato late blight" in en["text"]
    assert "KVK" in en["disclaimer"]
    hi = advisor.advise("Potato___Late_blight", "hi", 0.95)
    assert "पछेती झुलसा" in hi["text"]


def test_advisor_unknown_label_raises():
    advisor = Advisor(KB, offline_llm())
    try:
        advisor.advise("Banana___Panama", "en")
    except KeyError:
        return
    raise AssertionError("expected KeyError")


# ---------- RAG chat ----------
def test_retriever_pins_detected_label():
    retriever = Retriever(KB, resolve_path("knowledge_base/docs"))
    notes = retriever.search("neem oil", k=3, label="Tomato___Early_blight")
    assert notes[0].source == "Tomato___Early_blight"
    assert len(notes) == 3


def test_chat_offline_answer_uses_notes():
    llm = offline_llm()
    chat = ChatAssistant(KB, Retriever(KB, resolve_path("knowledge_base/docs")), llm)
    answer = chat.answer("How do I control whitefly?", "en", None)
    assert answer["source"] == "static"
    assert answer["references"]


# ---------- TTS text cleaning ----------
def test_clean_for_speech_removes_markdown():
    assert clean_for_speech("**Bold** # head\n- item") == "Bold head\nitem"
