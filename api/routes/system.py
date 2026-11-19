"""Health check, labels list, feedback and anonymous usage statistics."""

from fastapi import APIRouter, Request

from api.limiter import PREDICT_LIMIT, limiter
from api.schemas import FeedbackRequest, FeedbackResponse, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = request.app.state
    loaded = state.predictor is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        version=state.cfg["project"]["version"],
        model_loaded=loaded,
        llm_enabled=state.llm.enabled,
    )


@router.get("/labels")
async def labels(request: Request) -> list[dict]:
    """All diseases the app knows, with English + Hindi names (used for the feedback form)."""
    kb = request.app.state.kb
    out = []
    for label in kb.labels():
        info = kb.get(label)
        out.append({"label": label, "crop_en": info["crop_en"], "crop_hi": info["crop_hi"],
                    "name_en": info["name_en"], "name_hi": info["name_hi"]})
    return out


@router.post("/feedback", response_model=FeedbackResponse)
@limiter.limit(PREDICT_LIMIT)
async def feedback(request: Request, body: FeedbackRequest) -> FeedbackResponse:
    fid = request.app.state.db.add_feedback(
        body.scan_id, body.predicted_label, body.correct, body.actual_label, body.comment
    )
    return FeedbackResponse(id=fid, message="Thank you! Your feedback helps improve KrishiMitra.")


@router.get("/stats")
async def stats(request: Request) -> dict:
    """Anonymous, aggregated usage numbers (no personal data)."""
    return request.app.state.db.stats()
