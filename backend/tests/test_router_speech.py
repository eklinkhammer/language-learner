"""Tests for app.routers.speech — speech analysis endpoint."""

from unittest.mock import AsyncMock, patch

import pytest


# ===== POST /api/speech/analyze =====


class TestAnalyzeSpeech:
    """Speech analysis endpoint tests. All 4 service calls are mocked."""

    @pytest.mark.anyio
    @patch("app.routers.speech.feedback.generate_feedback", new_callable=AsyncMock)
    @patch("app.routers.speech.phoneme.compute_gop_scores", new_callable=AsyncMock)
    @patch("app.routers.speech.phoneme.text_to_phonemes", new_callable=AsyncMock)
    @patch("app.routers.speech.whisper.transcribe", new_callable=AsyncMock)
    async def test_success(self, mock_transcribe, mock_t2p, mock_gop, mock_feedback, client):
        """Success → 200 with full SpeechAnalysisResponse."""
        mock_transcribe.return_value = {"text": "hola", "segments": [], "language": "es"}
        mock_t2p.return_value = ["o", "l", "a"]
        mock_gop.return_value = [
            {"phoneme": "o", "expected": "o", "score": 0.9, "is_correct": True},
            {"phoneme": "l", "expected": "l", "score": 0.8, "is_correct": True},
            {"phoneme": "a", "expected": "a", "score": 0.85, "is_correct": True},
        ]
        mock_feedback.return_value = (["Excellent pronunciation!"], [])

        response = await client.post(
            "/api/speech/analyze",
            data={"expected_text": "hola", "language": "es"},
            files={"audio": ("test.wav", b"fake audio data", "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "hola"
        assert data["expected_text"] == "hola"
        assert len(data["phoneme_scores"]) == 3
        assert data["overall_score"] == 1.0
        assert "Excellent" in data["feedback"][0]

    @pytest.mark.anyio
    async def test_unsupported_language(self, client):
        """Unsupported language → 422."""
        response = await client.post(
            "/api/speech/analyze",
            data={"expected_text": "test", "language": "xx"},
            files={"audio": ("test.wav", b"data", "audio/wav")},
        )
        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_bad_content_type(self, client):
        """Bad content-type → 415."""
        response = await client.post(
            "/api/speech/analyze",
            data={"expected_text": "test", "language": "es"},
            files={"audio": ("test.txt", b"data", "text/plain")},
        )
        assert response.status_code == 415

    @pytest.mark.anyio
    async def test_file_too_large(self, client):
        """File too large (26 MB) → 413."""
        big_data = b"x" * (26 * 1024 * 1024)
        response = await client.post(
            "/api/speech/analyze",
            data={"expected_text": "test", "language": "es"},
            files={"audio": ("test.wav", big_data, "audio/wav")},
        )
        assert response.status_code == 413

    @pytest.mark.anyio
    @patch("app.routers.speech.whisper.transcribe", new_callable=AsyncMock)
    async def test_pipeline_error(self, mock_transcribe, client):
        """Pipeline error → 500."""
        mock_transcribe.side_effect = RuntimeError("model crashed")

        response = await client.post(
            "/api/speech/analyze",
            data={"expected_text": "test", "language": "es"},
            files={"audio": ("test.wav", b"data", "audio/wav")},
        )
        assert response.status_code == 500

    @pytest.mark.anyio
    @patch("app.routers.speech.Path")
    @patch("app.routers.speech.feedback.generate_feedback", new_callable=AsyncMock)
    @patch("app.routers.speech.phoneme.compute_gop_scores", new_callable=AsyncMock)
    @patch("app.routers.speech.phoneme.text_to_phonemes", new_callable=AsyncMock)
    @patch("app.routers.speech.whisper.transcribe", new_callable=AsyncMock)
    async def test_temp_file_cleanup(self, mock_transcribe, mock_t2p, mock_gop,
                                     mock_feedback, mock_path, client):
        """Temp file is cleaned up via Path.unlink."""
        mock_transcribe.return_value = {"text": "hola", "segments": [], "language": "es"}
        mock_t2p.return_value = ["o"]
        mock_gop.return_value = [
            {"phoneme": "o", "expected": "o", "score": 0.9, "is_correct": True},
        ]
        mock_feedback.return_value = (["Good!"], [])

        response = await client.post(
            "/api/speech/analyze",
            data={"expected_text": "hola", "language": "es"},
            files={"audio": ("test.wav", b"data", "audio/wav")},
        )
        assert response.status_code == 200
