import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.models.schemas import (
    Exercise,
    ExerciseEvalResponse,
    ExerciseListResponse,
    SpeechAnalysisResponse,
    PhonemeScore,
)
from app.services import whisper, phoneme, feedback

router = APIRouter()

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
    exercise = next((e for e in STUB_EXERCISES if e.id == exercise_id), STUB_EXERCISES[0])

    # Save uploaded audio to temp file
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

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
    overall_score = (
        sum(r["score"] for r in gop_results) / len(gop_results)
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
            overall_score=round(overall_score, 3),
            feedback=feedback_lines,
        ),
    )
