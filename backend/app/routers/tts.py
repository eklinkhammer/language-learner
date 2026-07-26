import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.config import settings
from app.services.tts import synthesize

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("")
async def get_tts(
    text: str = Query(..., min_length=1, max_length=500),
    language: str = Query(...),
):
    """Generate TTS audio for the given text and language."""
    if language not in settings.supported_languages:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language: {language}. Supported: {settings.supported_languages}",
        )

    try:
        path = await synthesize(text, language)
    except Exception as exc:
        logger.exception("TTS synthesis failed for text=%r language=%s", text, language)
        raise HTTPException(
            status_code=503,
            detail="Text-to-speech service is temporarily unavailable. Please try again later.",
        ) from exc

    return FileResponse(
        path,
        media_type="audio/mpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )
