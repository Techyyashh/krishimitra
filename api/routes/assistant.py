"""AI assistant endpoints: advice, chat and text-to-speech."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from api.limiter import LLM_LIMIT, limiter
from api.schemas import AdviceRequest, AdviceResponse, ChatRequest, ChatResponse, TTSRequest
from src.speech.tts import TTSError, synthesize

router = APIRouter(tags=["assistant"])


@router.post("/advice", response_model=AdviceResponse)
@limiter.limit(LLM_LIMIT)
async def advice(request: Request, body: AdviceRequest) -> AdviceResponse:
    state = request.app.state
    if not state.kb.has(body.label):
        raise HTTPException(404, f"Unknown label: {body.label}")
    result = await run_in_threadpool(state.advisor.advise, body.label, body.lang, body.confidence)
    return AdviceResponse(label=body.label, lang=body.lang, **result)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(LLM_LIMIT)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    result = await run_in_threadpool(request.app.state.chat.answer, body.question, body.lang, body.label)
    return ChatResponse(**result)


@router.post("/tts", response_class=Response, responses={200: {"content": {"audio/mpeg": {}}}})
@limiter.limit(LLM_LIMIT)
async def tts(request: Request, body: TTSRequest) -> Response:
    try:
        audio = await run_in_threadpool(synthesize, body.text, body.lang)
    except TTSError as err:
        raise HTTPException(503, f"Voice is not available right now: {err}") from err
    return Response(content=audio, media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=86400"})
