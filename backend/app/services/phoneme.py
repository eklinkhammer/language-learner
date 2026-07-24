"""Phoneme alignment and GOP scoring using wav2vec2."""


async def align_phonemes(audio_path: str, transcript: str, language: str = "es") -> list[dict]:
    """Align phonemes from audio using wav2vec2.

    Args:
        audio_path: Path to the audio file.
        transcript: Expected transcript text.
        language: Language code.

    Returns:
        List of dicts with phoneme, start_time, end_time, and confidence.
    """
    # TODO: Implement with transformers wav2vec2
    # from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    # processor = Wav2Vec2Processor.from_pretrained(settings.wav2vec2_model)
    # model = Wav2Vec2ForCTC.from_pretrained(settings.wav2vec2_model)
    # ... run forced alignment and extract phoneme-level posteriors
    return []


async def compute_gop_scores(
    audio_path: str, expected_phonemes: list[str], aligned_phonemes: list[dict]
) -> list[dict]:
    """Compute Goodness of Pronunciation (GOP) scores.

    Compares expected phonemes to actual pronunciation using
    posterior probabilities from the wav2vec2 model.

    Args:
        audio_path: Path to the audio file.
        expected_phonemes: List of expected IPA phonemes.
        aligned_phonemes: Output from align_phonemes.

    Returns:
        List of dicts with phoneme, expected, score, and is_correct.
    """
    # TODO: Implement GOP scoring
    # GOP = log P(phoneme | audio_segment) — compare expected vs actual
    return []
