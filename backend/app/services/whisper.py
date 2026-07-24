"""Whisper transcription service using MLX on Apple Silicon."""


async def transcribe(audio_path: str, language: str = "es") -> dict:
    """Transcribe audio file using mlx-whisper.

    Args:
        audio_path: Path to the audio file.
        language: Language code (e.g., "es", "de", "zh").

    Returns:
        Dict with "text" (full transcript) and "segments" (timestamped segments).
    """
    # TODO: Implement with mlx_whisper
    # import mlx_whisper
    # result = mlx_whisper.transcribe(
    #     audio_path,
    #     path_or_hf_repo=settings.whisper_model,
    #     language=language,
    # )
    # return result
    return {
        "text": "(stub) transcription placeholder",
        "segments": [],
    }
