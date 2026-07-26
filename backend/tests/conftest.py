"""Shared test fixtures and module patching for the language-learner backend."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Pre-patch mlx_whisper before any app modules can import it.
# mlx_whisper may not be installable in CI (Apple-Silicon only).
# ---------------------------------------------------------------------------
if "mlx_whisper" not in sys.modules:
    sys.modules["mlx_whisper"] = MagicMock()


from app.main import app  # noqa: E402


# ---------------------------------------------------------------------------
# Force asyncio backend only (skip trio)
# ---------------------------------------------------------------------------

@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


# ---------------------------------------------------------------------------
# ASGI test client
# ---------------------------------------------------------------------------

@pytest.fixture
async def client():
    """Async httpx ASGI test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Synthetic tensors for phoneme tests
# ---------------------------------------------------------------------------

@pytest.fixture
def small_vocab():
    """Minimal vocabulary mapping token→id."""
    return {"<pad>": 0, "a": 1, "b": 2, "c": 3, "d": 4}


@pytest.fixture
def small_id_to_token(small_vocab):
    """Reverse vocabulary mapping id→token."""
    return {v: k for k, v in small_vocab.items()}


@pytest.fixture
def frame_duration():
    """Frame duration in seconds (20 ms per frame)."""
    return 0.02


@pytest.fixture
def small_log_probs():
    """Synthetic (T=10, V=5) log-prob tensor.

    Greedy decode should yield: "a" (frames 0-4), "b" (frames 5-9).
    Token IDs: <pad>=0, a=1, b=2, c=3, d=4.
    """
    T, V = 10, 5
    # Start with uniform low probs
    lp = torch.full((T, V), -10.0)
    # Frames 0-4: peak at token 1 ("a")
    lp[0:5, 1] = -0.1
    # Frames 5-9: peak at token 2 ("b")
    lp[5:10, 2] = -0.1
    return lp


# ---------------------------------------------------------------------------
# Sample GOP scores for feedback tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_gop_scores_mixed():
    """GOP scores with a mix of correct and incorrect phonemes."""
    return [
        {"phoneme": "a", "expected": "a", "score": 0.9, "is_correct": True},
        {"phoneme": "s", "expected": "s", "score": 0.8, "is_correct": True},
        {"phoneme": "t", "expected": "d", "score": 0.1, "is_correct": False},
        {"phoneme": "∅", "expected": "o", "score": 0.05, "is_correct": False},
    ]


@pytest.fixture
def sample_gop_scores_all_correct():
    """GOP scores where all phonemes are correct."""
    return [
        {"phoneme": "a", "expected": "a", "score": 0.9, "is_correct": True},
        {"phoneme": "b", "expected": "b", "score": 0.85, "is_correct": True},
        {"phoneme": "c", "expected": "c", "score": 0.7, "is_correct": True},
    ]


@pytest.fixture
def sample_gop_scores_with_silence():
    """GOP scores with multiple silence/undetected phonemes."""
    return [
        {"phoneme": "a", "expected": "a", "score": 0.9, "is_correct": True},
        {"phoneme": "∅", "expected": "b", "score": 0.02, "is_correct": False},
        {"phoneme": "∅", "expected": "c", "score": 0.01, "is_correct": False},
    ]
