"""Tests for app.services.llm — LLM tutor service."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm import (
    LLMServiceError,
    _build_prompt,
    _default_suggestions,
    _parse_suggestions,
)


# ===== _parse_suggestions() =====


class TestParseSuggestions:
    """SUGGESTIONS line parsing tests."""

    def test_valid_json_array(self):
        text = 'Hello!\nSUGGESTIONS: ["one", "two", "three"]'
        reply, suggestions = _parse_suggestions(text)
        assert reply == "Hello!"
        assert suggestions == ["one", "two", "three"]

    def test_no_suggestions_line(self):
        text = "Just a normal reply.\nNothing special."
        reply, suggestions = _parse_suggestions(text)
        assert reply == "Just a normal reply.\nNothing special."
        assert suggestions == []

    def test_invalid_json_treated_as_text(self):
        text = "Hello!\nSUGGESTIONS: not valid json"
        reply, suggestions = _parse_suggestions(text)
        assert "SUGGESTIONS: not valid json" in reply
        assert suggestions == []

    def test_empty_array(self):
        text = "Hello!\nSUGGESTIONS: []"
        reply, suggestions = _parse_suggestions(text)
        assert reply == "Hello!"
        assert suggestions == []

    def test_multiple_suggestions_lines_merged(self):
        text = (
            "Hello!\n"
            'SUGGESTIONS: ["a", "b"]\n'
            "More text.\n"
            'SUGGESTIONS: ["c"]'
        )
        reply, suggestions = _parse_suggestions(text)
        assert suggestions == ["a", "b", "c"]
        assert "More text." in reply


# ===== _build_prompt() =====


class TestBuildPrompt:
    """Prompt construction tests."""

    def test_no_history(self):
        result = _build_prompt("Hello", "es", [])
        assert result == "Learner: Hello\nTutor:"

    def test_with_alternating_history(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hola"},
        ]
        result = _build_prompt("How are you?", "es", history)
        lines = result.split("\n")
        assert lines[0] == "Learner: Hi"
        assert lines[1] == "Tutor: Hola"
        assert lines[2] == "Learner: How are you?"
        assert lines[3] == "Tutor:"

    def test_correct_role_mapping(self):
        history = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        result = _build_prompt("D", "es", history)
        assert "Learner: A" in result
        assert "Tutor: B" in result
        assert "Learner: C" in result
        assert result.endswith("Learner: D\nTutor:")


# ===== chat() =====


class TestChat:
    """Async chat tests (mock subprocess)."""

    @pytest.mark.anyio
    @patch("app.services.llm._find_claude", return_value="claude")
    async def test_successful_response(self, mock_find):
        from app.services.llm import chat

        response_json = json.dumps(
            {"result": 'Hola! ¿Cómo estás?\nSUGGESTIONS: ["Bien", "Mal"]'}
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(response_json.encode(), b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await chat("Hello", "es", [])

        assert "reply" in result
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0

    @pytest.mark.anyio
    async def test_unsupported_language(self):
        from app.services.llm import chat

        with pytest.raises(LLMServiceError, match="Unsupported language"):
            await chat("Hello", "xx", [])

    @pytest.mark.anyio
    @patch("app.services.llm._find_claude", return_value="claude")
    async def test_timeout(self, mock_find):
        from app.services.llm import chat

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(LLMServiceError, match="timed out"):
                await chat("Hello", "es", [])


# ===== _default_suggestions() =====


class TestDefaultSuggestions:
    """Fallback suggestions tests."""

    def test_each_language_returns_3_strings(self):
        for lang in ("es", "hr", "de", "zh"):
            result = _default_suggestions(lang)
            assert len(result) == 3
            assert all(isinstance(s, str) for s in result)

    def test_unknown_falls_back_to_spanish(self):
        result = _default_suggestions("xx")
        assert result == _default_suggestions("es")
