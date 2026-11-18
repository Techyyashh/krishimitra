"""Request / response models. Pydantic validates every input automatically (→ HTTP 422 on bad data)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.llm.knowledge import LANGUAGES


def _check_lang(value: str) -> str:
    if value not in LANGUAGES:
        raise ValueError(f"Unsupported language '{value}'. Use one of: {', '.join(LANGUAGES)}")
    return value


class PredictionItem(BaseModel):
    label: str
    crop: str
    crop_hi: str | None = None
    disease: str
    disease_hi: str | None = None
    is_healthy: bool
    confidence: float = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    status: Literal["ok", "low_confidence", "not_leaf"]
    message: str
    prediction: PredictionItem | None = None
    top: list[PredictionItem] = []
    info: dict[str, Any] | None = None
    leaf_score: float
    inference_ms: float | None = None
    scan_id: int | None = None


class AdviceRequest(BaseModel):
    label: str = Field(min_length=3, max_length=120)
    lang: str = "en"
    confidence: float = Field(default=1.0, ge=0, le=1)

    @field_validator("lang")
    @classmethod
    def validate_lang(cls, value: str) -> str:
        return _check_lang(value)


class AdviceResponse(BaseModel):
    label: str
    lang: str
    source: Literal["ai", "cache", "static"]
    text: str
    disclaimer: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    lang: str = "en"
    label: str | None = Field(default=None, max_length=120)

    @field_validator("lang")
    @classmethod
    def validate_lang(cls, value: str) -> str:
        return _check_lang(value)


class ChatResponse(BaseModel):
    answer: str
    source: Literal["ai", "static"]
    references: list[str]


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    lang: str = "en"

    @field_validator("lang")
    @classmethod
    def validate_lang(cls, value: str) -> str:
        return _check_lang(value)


class FeedbackRequest(BaseModel):
    scan_id: int | None = None
    predicted_label: str = Field(max_length=120)
    correct: bool
    actual_label: str | None = Field(default=None, max_length=120)
    comment: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    id: int
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    llm_enabled: bool
