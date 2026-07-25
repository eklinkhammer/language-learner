from typing import Literal

from pydantic import BaseModel


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
    content: str


class ChatRequest(BaseModel):
    message: str
    language: str = "es"
    history: list[ChatMessage] = []


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
