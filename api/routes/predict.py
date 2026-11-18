"""POST /api/predict – photo in, disease + info card out."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.limiter import PREDICT_LIMIT, limiter
from api.schemas import PredictionItem, PredictResponse
from api.security import read_image_upload
from src.inference.leaf_validator import is_probably_leaf

router = APIRouter(tags=["prediction"])


def _enrich(item: dict, kb) -> PredictionItem:
    """Add Hindi crop / disease names from the knowledge base."""
    extra = {}
    if kb.has(item["label"]):
        info = kb.get(item["label"])
        extra = {"crop_hi": info["crop_hi"], "disease_hi": info["name_hi"]}
    return PredictionItem(**item, **extra)


@router.post("/predict", response_model=PredictResponse)
@limiter.limit(PREDICT_LIMIT)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictResponse:
    state = request.app.state
    if state.predictor is None:
        raise HTTPException(503, "The AI model is not loaded yet. Please try again later.")
    cfg = state.cfg
    image = await read_image_upload(file, cfg["api"]["max_upload_mb"] * 1024 * 1024)

    looks_like_leaf, leaf_score = is_probably_leaf(image, cfg["inference"]["leaf_min_ratio"])
    if not looks_like_leaf:
        scan_id = state.db.log_scan(None, None, "not_leaf")
        return PredictResponse(
            status="not_leaf", leaf_score=leaf_score, scan_id=scan_id,
            message="This does not look like a leaf. Take a close photo of ONE leaf in daylight.",
        )

    result = await run_in_threadpool(state.predictor.predict, image)
    top = [_enrich(item, state.kb) for item in result["top"]]
    best = top[0]
    status = "ok" if result["confident"] else "low_confidence"
    message = (
        "Prediction ready." if status == "ok"
        else "Not sure. Retake the photo: one leaf, in daylight, filling the screen, not blurry."
    )
    info = state.kb.get(best.label) if state.kb.has(best.label) else None
    scan_id = state.db.log_scan(best.label, best.confidence, status)
    return PredictResponse(
        status=status, message=message, prediction=best, top=top, info=info,
        leaf_score=leaf_score, inference_ms=result["inference_ms"], scan_id=scan_id,
    )
