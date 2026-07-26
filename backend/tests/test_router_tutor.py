"""Tests for app.routers.tutor — language tutor chat endpoint."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.llm import LLMServiceError


# ===== POST /api/tutor/chat =====


class TestTutorChat:
    """Tutor chat endpoint tests."""

    @pytest.mark.anyio
    @patch("app.services.llm.chat", new_callable=AsyncMock)
    async def test_success(self, mock_chat, client):
        """Success → 200 with reply + suggestions."""
        mock_chat.return_value = {
            "reply": "¡Hola! ¿Cómo estás?",
            "suggestions": ["Bien", "Mal", "Regular"],
        }

        response = await client.post(
            "/api/tutor/chat",
            json={"message": "Hello", "language": "es", "history": []},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == "¡Hola! ¿Cómo estás?"
        assert len(data["suggestions"]) == 3

    @pytest.mark.anyio
    @patch("app.services.llm.chat", new_callable=AsyncMock)
    async def test_llm_service_error(self, mock_chat, client):
        """LLMServiceError → 503."""
        mock_chat.side_effect = LLMServiceError("claude CLI not found")

        response = await client.post(
            "/api/tutor/chat",
            json={"message": "Hello", "language": "es", "history": []},
        )
        assert response.status_code == 503

    @pytest.mark.anyio
    async def test_empty_message(self, client):
        """Empty message → 422 (Pydantic min_length=1)."""
        response = await client.post(
            "/api/tutor/chat",
            json={"message": "", "language": "es", "history": []},
        )
        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_history_too_long(self, client):
        """History with 51 items → 422 (Pydantic max_length=50)."""
        long_history = [
            {"role": "user", "content": f"msg {i}"} for i in range(51)
        ]
        response = await client.post(
            "/api/tutor/chat",
            json={"message": "Hello", "language": "es", "history": long_history},
        )
        assert response.status_code == 422
