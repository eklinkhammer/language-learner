"""Tests for app.routers.exercises — exercise listing and evaluation."""

from unittest.mock import AsyncMock, patch

import pytest


# ===== GET /api/exercises =====


class TestListExercises:
    """Exercise listing endpoint tests."""

    @pytest.mark.anyio
    async def test_default_spanish(self, client):
        """Default → 3 Spanish exercises."""
        response = await client.get("/api/exercises?language=es")
        assert response.status_code == 200
        data = response.json()
        assert len(data["exercises"]) == 3

    @pytest.mark.anyio
    async def test_filter_by_difficulty(self, client):
        """Filter by difficulty=beginner → 2 exercises."""
        response = await client.get("/api/exercises?language=es&difficulty=beginner")
        assert response.status_code == 200
        data = response.json()
        assert len(data["exercises"]) == 2
        assert all(e["difficulty"] == "beginner" for e in data["exercises"])

    @pytest.mark.anyio
    async def test_nonexistent_language(self, client):
        """Nonexistent language → empty list."""
        response = await client.get("/api/exercises?language=xx")
        assert response.status_code == 200
        data = response.json()
        assert data["exercises"] == []


# ===== POST /api/exercises/evaluate =====


class TestEvaluateExercise:
    """Exercise evaluation endpoint tests."""

    @pytest.mark.anyio
    @patch("app.routers.exercises.feedback.generate_feedback", new_callable=AsyncMock)
    @patch("app.routers.exercises.phoneme.compute_gop_scores", new_callable=AsyncMock)
    @patch("app.routers.exercises.phoneme.text_to_phonemes", new_callable=AsyncMock)
    @patch("app.routers.exercises.whisper.transcribe", new_callable=AsyncMock)
    async def test_success(self, mock_transcribe, mock_t2p, mock_gop, mock_feedback, client):
        """Success → 200 with exercise + analysis."""
        mock_transcribe.return_value = {"text": "hola", "segments": [], "language": "es"}
        mock_t2p.return_value = ["o", "l", "a"]
        mock_gop.return_value = [
            {"phoneme": "o", "expected": "o", "score": 0.9, "is_correct": True},
        ]
        mock_feedback.return_value = (["Great!"], [])

        response = await client.post(
            "/api/exercises/evaluate",
            data={"exercise_id": "es-001", "language": "es"},
            files={"audio": ("test.wav", b"fake audio", "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "exercise" in data
        assert "analysis" in data
        assert data["exercise"]["id"] == "es-001"

    @pytest.mark.anyio
    async def test_unknown_exercise_id(self, client):
        """Unknown exercise_id → 404."""
        response = await client.post(
            "/api/exercises/evaluate",
            data={"exercise_id": "xx-999", "language": "es"},
            files={"audio": ("test.wav", b"data", "audio/wav")},
        )
        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_unsupported_language(self, client):
        """Unsupported language → 422."""
        response = await client.post(
            "/api/exercises/evaluate",
            data={"exercise_id": "es-001", "language": "xx"},
            files={"audio": ("test.wav", b"data", "audio/wav")},
        )
        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_bad_content_type(self, client):
        """Bad content-type → 415."""
        response = await client.post(
            "/api/exercises/evaluate",
            data={"exercise_id": "es-001", "language": "es"},
            files={"audio": ("test.txt", b"data", "text/plain")},
        )
        assert response.status_code == 415

    @pytest.mark.anyio
    @patch("app.routers.exercises.whisper.transcribe", new_callable=AsyncMock)
    async def test_pipeline_error(self, mock_transcribe, client):
        """Pipeline error → 500."""
        mock_transcribe.side_effect = RuntimeError("boom")

        response = await client.post(
            "/api/exercises/evaluate",
            data={"exercise_id": "es-001", "language": "es"},
            files={"audio": ("test.wav", b"data", "audio/wav")},
        )
        assert response.status_code == 500
