from fastapi import APIRouter, File, Form, UploadFile

from app.models.schemas import SpeechAnalysisResponse, PhonemeScore

router = APIRouter()


@router.post("/analyze", response_model=SpeechAnalysisResponse)
async def analyze_speech(
    audio: UploadFile = File(...),
    expected_text: str = Form(...),
    language: str = Form("es"),
):
    """Analyze pronunciation of uploaded audio against expected text.

    Pipeline: audio → Whisper transcription → wav2vec2 phoneme alignment →
    GOP scoring → epitran/panphon feedback
    """
    # TODO: Implement full pipeline
    # 1. Save uploaded audio to temp file
    # 2. Transcribe with Whisper
    # 3. Get phoneme alignment with wav2vec2
    # 4. Compute GOP scores
    # 5. Generate feedback with epitran/panphon

    return SpeechAnalysisResponse(
        transcript="(stub) hola mundo",
        expected_text=expected_text,
        phoneme_scores=[
            PhonemeScore(phoneme="o", expected="o", score=0.95, is_correct=True),
            PhonemeScore(phoneme="l", expected="l", score=0.90, is_correct=True),
            PhonemeScore(phoneme="a", expected="a", score=0.85, is_correct=True),
        ],
        overall_score=0.90,
        feedback=["(stub) Good pronunciation overall!"],
    )
