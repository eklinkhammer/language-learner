"""Whisper transcription service using MLX on Apple Silicon."""

import asyncio
import logging
import subprocess
from pathlib import Path

import mlx_whisper

from app.config import settings

log = logging.getLogger(__name__)

# Language code to Whisper language name mapping
LANGUAGE_MAP = {
    "es": "es",
    "hr": "hr",
    "de": "de",
    "zh": "zh",
}


def _get_model_repo() -> str:
    """Return the configured Whisper model repo identifier."""
    return settings.whisper_model


def _convert_to_wav(input_path: str) -> str:
    """Convert audio file to 16kHz mono WAV using ffmpeg.

    Whisper expects 16kHz mono audio. Browser MediaRecorder typically
    produces webm/opus which needs conversion.
    """
    wav_path = input_path + ".wav"
    try:
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
            timeout=60,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timed out converting %s", input_path)
        raise RuntimeError("Audio conversion timed out")
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg failed: %s", e.stderr.decode(errors="replace"))
        raise RuntimeError(f"Audio conversion failed: {e.stderr.decode(errors='replace')[:500]}")
    return wav_path


def _transcribe_sync(audio_path: str, language: str) -> dict:
    """Synchronous transcription — runs in thread pool."""
    model_repo = _get_model_repo()

    lang = LANGUAGE_MAP.get(language)
    if lang is None:
        raise ValueError(
            f"Unsupported language: {language!r}. "
            f"Supported: {list(LANGUAGE_MAP.keys())}"
        )

    wav_path = None
    # Convert to WAV if not already a supported format
    path = Path(audio_path)
    if path.suffix.lower() not in (".wav", ".mp3", ".flac", ".m4a"):
        wav_path = _convert_to_wav(audio_path)
        audio_path = wav_path

    try:
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_repo,
            language=lang,
            word_timestamps=True,
            verbose=False,
        )

        log.info(
            "Transcribed %s (%s): %d chars, %d segments",
            audio_path, lang, len(result["text"]), len(result.get("segments", [])),
        )

        return {
            "text": result["text"].strip(),
            "segments": result.get("segments", []),
            "language": result.get("language", lang),
        }
    finally:
        if wav_path is not None:
            Path(wav_path).unlink(missing_ok=True)


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
