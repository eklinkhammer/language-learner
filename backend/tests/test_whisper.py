"""Tests for app.services.whisper — Whisper transcription service."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest


# ===== _convert_to_wav() =====


class TestConvertToWav:
    """ffmpeg conversion tests (mock subprocess.run)."""

    @patch("app.services.whisper.subprocess.run")
    def test_success(self, mock_run):
        """Correct ffmpeg args, returns .wav path."""
        from app.services.whisper import _convert_to_wav

        mock_run.return_value = MagicMock(returncode=0)
        result = _convert_to_wav("/tmp/audio.webm")

        assert result == "/tmp/audio.webm.wav"
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-ar" in args
        assert "16000" in args

    @patch("app.services.whisper.subprocess.run", side_effect=FileNotFoundError)
    def test_ffmpeg_not_found(self, mock_run):
        """FileNotFoundError → RuntimeError 'ffmpeg is not installed'."""
        from app.services.whisper import _convert_to_wav

        with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
            _convert_to_wav("/tmp/audio.webm")

    @patch("app.services.whisper.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=60))
    def test_timeout(self, mock_run):
        """TimeoutExpired → RuntimeError 'timed out'."""
        from app.services.whisper import _convert_to_wav

        with pytest.raises(RuntimeError, match="timed out"):
            _convert_to_wav("/tmp/audio.webm")

    @patch("app.services.whisper.subprocess.run")
    def test_called_process_error(self, mock_run):
        """CalledProcessError → RuntimeError with stderr."""
        from app.services.whisper import _convert_to_wav

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"some error"
        )
        with pytest.raises(RuntimeError, match="some error"):
            _convert_to_wav("/tmp/audio.webm")


# ===== _transcribe_sync() =====


class TestTranscribeSync:
    """Transcription tests (mock mlx_whisper)."""

    @patch("app.services.whisper.mlx_whisper")
    def test_wav_input_skips_conversion(self, mock_mlx):
        """WAV input skips ffmpeg conversion."""
        from app.services.whisper import _transcribe_sync

        mock_mlx.transcribe.return_value = {
            "text": " hello ",
            "segments": [],
            "language": "es",
        }

        result = _transcribe_sync("/tmp/audio.wav", "es")
        assert result["text"] == "hello"
        mock_mlx.transcribe.assert_called_once()

    @patch("app.services.whisper._convert_to_wav")
    @patch("app.services.whisper.mlx_whisper")
    @patch("app.services.whisper.Path")
    def test_webm_triggers_conversion_and_cleanup(self, mock_path_cls, mock_mlx, mock_convert):
        """webm input triggers conversion + cleanup."""
        from app.services.whisper import _transcribe_sync

        mock_convert.return_value = "/tmp/audio.webm.wav"
        mock_mlx.transcribe.return_value = {
            "text": " hello ",
            "segments": [],
            "language": "es",
        }
        mock_path_instance = MagicMock()
        mock_path_cls.return_value = mock_path_instance
        mock_path_cls.return_value.suffix.lower.return_value = ".webm"

        result = _transcribe_sync("/tmp/audio.webm", "es")
        assert result["text"] == "hello"
        mock_convert.assert_called_once()

    @patch("app.services.whisper.mlx_whisper")
    def test_unsupported_language(self, mock_mlx):
        """Unsupported language → ValueError."""
        from app.services.whisper import _transcribe_sync

        with pytest.raises(ValueError, match="Unsupported language"):
            _transcribe_sync("/tmp/audio.wav", "xx")

    @patch("app.services.whisper._convert_to_wav")
    @patch("app.services.whisper.mlx_whisper")
    @patch("app.services.whisper.Path")
    def test_cleanup_on_transcription_error(self, mock_path_cls, mock_mlx, mock_convert):
        """Temp WAV cleaned up even on transcription error."""
        from app.services.whisper import _transcribe_sync

        mock_convert.return_value = "/tmp/audio.webm.wav"
        mock_mlx.transcribe.side_effect = RuntimeError("model error")

        mock_path_instance = MagicMock()
        mock_path_cls.return_value = mock_path_instance
        mock_path_cls.return_value.suffix.lower.return_value = ".webm"

        with pytest.raises(RuntimeError, match="model error"):
            _transcribe_sync("/tmp/audio.webm", "es")
