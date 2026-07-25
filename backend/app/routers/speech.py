import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.models.schemas import SpeechAnalysisResponse, PhonemeScore
from app.services import whisper, phoneme, feedback

log = logging.getLogger(__name__)

router = APIRouter()

MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB
ALLOWED_AUDIO_TYPES = {"audio/webm", "audio/wav", "audio/mpeg", "audio/mp4", "audio/ogg", "audio/flac"}
ALLOWED_SUFFIXES = {".webm", ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".mp4"}


@router.post("/analyze", response_model=SpeechAnalysisResponse)
async def analyze_speech(
    audio: UploadFile = File(...),
    expected_text: str = Form(...),
    language: str = Form("es"),
):
    """Analyze pronunciation of uploaded audio against expected text.

    Pipeline: audio → Whisper transcription → epitran G2P →
    wav2vec2 forced alignment → GOP scoring → panphon feedback
    """
    if language not in settings.supported_languages:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language: {language}. Supported: {settings.supported_languages}",
        )

    if audio.content_type and audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported audio type: {audio.content_type}")

    contents = await audio.read()
    if len(contents) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="Audio file too large (max 25 MB)")

    # Save uploaded audio to temp file
    suffix = Path(audio.filename or "audio.webm").suffix.lower() or ".webm"
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 1. Transcribe with Whisper
        result = await whisper.transcribe(tmp_path, language)
        transcript = result["text"]

        # 2. Convert expected text to model phonemes (epitran → vocab tokens)
        expected_phonemes = await phoneme.text_to_phonemes(expected_text, language)

        # 3. Forced alignment + GOP scoring
        gop_results = await phoneme.compute_gop_scores(tmp_path, expected_phonemes)

        phoneme_scores = [
            PhonemeScore(
                phoneme=r["phoneme"],
                expected=r["expected"],
                score=r["score"],
                is_correct=r["is_correct"],
            )
            for r in gop_results
        ]

        # Use fraction of correct phonemes for a user-friendly score
        correct_fraction = (
            sum(1 for r in gop_results if r["is_correct"]) / len(gop_results)
            if gop_results
            else 0.0
        )

        # 4. Generate human-readable feedback with epitran/panphon
        feedback_lines = await feedback.generate_feedback(
            expected_text, transcript, gop_results, language
        )

        return SpeechAnalysisResponse(
            transcript=transcript,
            expected_text=expected_text,
            phoneme_scores=phoneme_scores,
            overall_score=round(correct_fraction, 3),
            feedback=feedback_lines,
        )
    except Exception:
        log.exception("Speech analysis pipeline failed")
        raise HTTPException(status_code=500, detail="Speech analysis failed. Please try again.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
