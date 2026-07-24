import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.models.schemas import SpeechAnalysisResponse, PhonemeScore
from app.services import whisper, phoneme, feedback

router = APIRouter()


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
    # Save uploaded audio to temp file
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

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

    overall_score = (
        sum(r["score"] for r in gop_results) / len(gop_results)
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
        overall_score=round(overall_score, 3),
        feedback=feedback_lines,
    )
