"""Whisper transcription service using MLX on Apple Silicon."""

import asyncio
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import mlx_whisper

from app.config import settings

# Language code to Whisper language name mapping
LANGUAGE_MAP = {
    "es": "es",
    "hr": "hr",
    "de": "de",
    "zh": "zh",
}


@lru_cache(maxsize=1)
def _warm_model() -> str:
    """Trigger model download/cache on first use. Returns the model path."""
    # mlx_whisper downloads and caches the model automatically on first transcribe call.
    # We just return the configured repo so callers don't need to know about config.
    return settings.whisper_model


def _convert_to_wav(input_path: str) -> str:
    """Convert audio file to 16kHz mono WAV using ffmpeg.

    Whisper expects 16kHz mono audio. Browser MediaRecorder typically
    produces webm/opus which needs conversion.
    """
    wav_path = input_path + ".wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            wav_path,
        ],
        capture_output=True,
        check=True,
    )
    return wav_path


def _transcribe_sync(audio_path: str, language: str) -> dict:
    """Synchronous transcription — runs in thread pool."""
    model_repo = _warm_model()
    lang = LANGUAGE_MAP.get(language, language)

    # Convert to WAV if not already
    path = Path(audio_path)
    if path.suffix not in (".wav", ".mp3", ".flac", ".m4a"):
        audio_path = _convert_to_wav(audio_path)

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
        language=lang,
        word_timestamps=True,
        verbose=False,
    )

    return {
        "text": result["text"].strip(),
        "segments": result.get("segments", []),
        "language": result.get("language", lang),
    }


async def transcribe(audio_path: str, language: str = "es") -> dict:
    """Transcribe audio file using mlx-whisper.

    Runs the synchronous MLX inference in a thread pool to avoid
    blocking the async event loop.

    Args:
        audio_path: Path to the audio file.
        language: Language code (e.g., "es", "de", "zh").

    Returns:
        Dict with "text" (full transcript), "segments" (timestamped segments),
        and "language".
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transcribe_sync, audio_path, language)
