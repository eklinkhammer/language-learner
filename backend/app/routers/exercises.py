from fastapi import APIRouter, File, Form, UploadFile

from app.models.schemas import (
    Exercise,
    ExerciseEvalResponse,
    ExerciseListResponse,
    SpeechAnalysisResponse,
    PhonemeScore,
)

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
    # TODO: Look up exercise, run speech analysis pipeline
    exercise = next((e for e in STUB_EXERCISES if e.id == exercise_id), STUB_EXERCISES[0])

    return ExerciseEvalResponse(
        exercise=exercise,
        analysis=SpeechAnalysisResponse(
            transcript=f"(stub) {exercise.phrase}",
            expected_text=exercise.phrase,
            phoneme_scores=[
                PhonemeScore(phoneme="stub", expected="stub", score=0.9, is_correct=True),
            ],
            overall_score=0.88,
            feedback=["(stub) Practice the rolling 'r' sound."],
        ),
    )
