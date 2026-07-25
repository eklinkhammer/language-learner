import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.models.schemas import (
    Exercise,
    ExerciseEvalResponse,
    ExerciseListResponse,
    SpeechAnalysisResponse,
    PhonemeScore,
)
from app.services import whisper, phoneme, feedback

log = logging.getLogger(__name__)

router = APIRouter()

MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB
ALLOWED_AUDIO_TYPES = {"audio/webm", "audio/wav", "audio/mpeg", "audio/mp4", "audio/ogg", "audio/flac"}
ALLOWED_SUFFIXES = {".webm", ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".mp4"}

# Stub exercises for MVP
STUB_EXERCISES: list[Exercise] = [
    Exercise(
        id="es-001",
        language="es",
        phrase="Hola, ¿cómo estás?",
        ipa="ˈola ˈkomo esˈtas",
        translation="Hello, how are you?",
        difficulty="beginner",
    ),
    Exercise(
        id="es-002",
        language="es",
        phrase="Buenos días",
        ipa="ˈbwenos ˈdias",
        translation="Good morning",
        difficulty="beginner",
    ),
    Exercise(
        id="es-003",
        language="es",
        phrase="Mucho gusto en conocerte",
        ipa="ˈmutʃo ˈɡusto en konoˈθeɾte",
        translation="Nice to meet you",
        difficulty="intermediate",
    ),
]


@router.get("", response_model=ExerciseListResponse)
async def list_exercises(language: str = "es", difficulty: str | None = None):
    """List available pronunciation exercises."""
    filtered = [e for e in STUB_EXERCISES if e.language == language]
    if difficulty:
        filtered = [e for e in filtered if e.difficulty == difficulty]
    return ExerciseListResponse(exercises=filtered)


@router.post("/evaluate", response_model=ExerciseEvalResponse)
async def evaluate_exercise(
    audio: UploadFile = File(...),
    exercise_id: str = Form(...),
    language: str = Form("es"),
):
    """Evaluate pronunciation for a specific exercise."""
    if language not in settings.supported_languages:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language: {language}. Supported: {settings.supported_languages}",
        )

    exercise = next((e for e in STUB_EXERCISES if e.id == exercise_id), None)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")

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
        # Transcribe with Whisper
        result = await whisper.transcribe(tmp_path, language)
        transcript = result["text"]

        # Convert exercise phrase to model phonemes and run GOP scoring
        expected_phonemes = await phoneme.text_to_phonemes(exercise.phrase, language)
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

        feedback_lines = await feedback.generate_feedback(
            exercise.phrase, transcript, gop_results, language
        )

        return ExerciseEvalResponse(
            exercise=exercise,
            analysis=SpeechAnalysisResponse(
                transcript=transcript,
                expected_text=exercise.phrase,
                phoneme_scores=phoneme_scores,
                overall_score=round(correct_fraction, 3),
                feedback=feedback_lines,
            ),
        )
    except HTTPException:
        raise
    except Exception:
        log.exception("Exercise evaluation pipeline failed")
        raise HTTPException(status_code=500, detail="Exercise evaluation failed. Please try again.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
