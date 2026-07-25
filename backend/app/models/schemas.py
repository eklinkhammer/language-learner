from typing import Literal

from pydantic import BaseModel, Field


class PhonemeScore(BaseModel):
    phoneme: str
    expected: str
    score: float
    is_correct: bool


class SpeechAnalysisResponse(BaseModel):
    transcript: str
    expected_text: str
    phoneme_scores: list[PhonemeScore]
    overall_score: float
    feedback: list[str]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    language: str = "es"
    history: list[ChatMessage] = Field(default=[], max_length=50)


class ChatResponse(BaseModel):
    reply: str
    suggestions: list[str] = []


class Exercise(BaseModel):
    id: str
    language: str
    phrase: str
    ipa: str
    translation: str
    difficulty: Literal["beginner", "intermediate", "advanced"]


class ExerciseListResponse(BaseModel):
    exercises: list[Exercise]


class ExerciseEvalRequest(BaseModel):
    exercise_id: str
    language: str = "es"


class ExerciseEvalResponse(BaseModel):
    exercise: Exercise
    analysis: SpeechAnalysisResponse
