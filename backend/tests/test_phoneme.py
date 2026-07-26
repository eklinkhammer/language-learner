"""Tests for app.services.phoneme — CTC alignment, GOP scoring, vocab mapping."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.services.phoneme import (
    _ctc_forced_align,
    _find_vocab_id,
    _greedy_decode,
)


# ===== _greedy_decode() =====


class TestGreedyDecode:
    """CTC greedy decoding tests."""

    def test_two_phonemes(self, small_log_probs, small_id_to_token, frame_duration):
        """Greedy decode yields 'a' then 'b' from synthetic tensor."""
        segments = _greedy_decode(small_log_probs, small_id_to_token, blank_id=0, frame_duration=frame_duration)
        phonemes = [s["phoneme"] for s in segments]
        assert phonemes == ["a", "b"]

    def test_all_blank(self, small_id_to_token, frame_duration):
        """All-blank input produces no segments."""
        T, V = 10, 5
        lp = torch.full((T, V), -10.0)
        lp[:, 0] = -0.01  # blank token wins every frame
        segments = _greedy_decode(lp, small_id_to_token, blank_id=0, frame_duration=frame_duration)
        assert segments == []

    def test_single_repeated_phoneme(self, small_id_to_token, frame_duration):
        """Single repeated phoneme collapses to 1 segment."""
        T, V = 10, 5
        lp = torch.full((T, V), -10.0)
        lp[:, 1] = -0.1  # token "a" wins every frame
        segments = _greedy_decode(lp, small_id_to_token, blank_id=0, frame_duration=frame_duration)
        assert len(segments) == 1
        assert segments[0]["phoneme"] == "a"

    def test_unknown_token_ids_skipped(self, frame_duration):
        """Token IDs not in id_to_token are skipped."""
        T, V = 6, 5
        lp = torch.full((T, V), -10.0)
        lp[0:3, 1] = -0.1  # token 1 wins
        lp[3:6, 4] = -0.1  # token 4 wins
        # Only map token 1, not token 4
        id_to_token = {0: "<pad>", 1: "a"}
        segments = _greedy_decode(lp, id_to_token, blank_id=0, frame_duration=frame_duration)
        phonemes = [s["phoneme"] for s in segments]
        assert phonemes == ["a"]

    def test_timestamps_monotonically_increasing(self, small_log_probs, small_id_to_token, frame_duration):
        """Start/end timestamps are monotonically increasing across segments."""
        segments = _greedy_decode(small_log_probs, small_id_to_token, blank_id=0, frame_duration=frame_duration)
        for i in range(len(segments)):
            assert segments[i]["start"] < segments[i]["end"]
            if i > 0:
                assert segments[i]["start"] >= segments[i - 1]["end"]


# ===== _ctc_forced_align() =====


class TestCtcForcedAlign:
    """CTC forced alignment (Viterbi DP) tests."""

    def test_two_target_alignment(self, small_log_probs):
        """Aligning two targets yields segments for both."""
        # target_ids = [1, 2] means "a", "b"
        segments = _ctc_forced_align(small_log_probs, target_ids=[1, 2], blank_id=0)
        assert len(segments) == 2
        target_indices = [s[0] for s in segments]
        assert target_indices == [0, 1]

    def test_empty_targets(self, small_log_probs):
        """Empty targets → empty result."""
        segments = _ctc_forced_align(small_log_probs, target_ids=[], blank_id=0)
        assert segments == []

    def test_t_less_than_2s_plus_1(self):
        """T < 2*S+1 → empty result (guard check)."""
        # T=4, S=3 → need T >= 7
        lp = torch.randn(4, 10)
        segments = _ctc_forced_align(lp, target_ids=[1, 2, 3], blank_id=0)
        assert segments == []

    def test_single_target(self):
        """Single target alignment produces exactly one segment."""
        T, V = 10, 5
        lp = torch.full((T, V), -10.0)
        lp[:, 1] = -0.5  # token "a" has reasonable prob everywhere
        lp[:, 0] = -2.0  # blank
        segments = _ctc_forced_align(lp, target_ids=[1], blank_id=0)
        assert len(segments) == 1
        assert segments[0][0] == 0  # target index 0

    def test_scores_are_negative_log_probs(self, small_log_probs):
        """Scores (avg_log_prob) are ≤ 0."""
        segments = _ctc_forced_align(small_log_probs, target_ids=[1, 2], blank_id=0)
        for seg in segments:
            assert seg[3] <= 0.0, f"avg_log_prob should be ≤ 0, got {seg[3]}"


# ===== _find_vocab_id() =====


class TestFindVocabId:
    """Vocabulary ID lookup tests."""

    def test_exact_match(self, small_vocab):
        assert _find_vocab_id("a", small_vocab) == 1

    def test_diacritic_stripping(self):
        vocab = {"a": 1, "b": 2}
        assert _find_vocab_id("aː", vocab) == 1

    def test_not_found(self, small_vocab):
        assert _find_vocab_id("z", small_vocab) is None

    def test_multiple_diacritics_stripped(self):
        vocab = {"a": 1}
        assert _find_vocab_id("aːˈ", vocab) == 1


# ===== _text_to_phonemes() =====


class TestTextToPhonemes:
    """Text-to-phoneme conversion tests (mock epitran + _load_model)."""

    @patch("app.services.phoneme._load_model")
    @patch("app.services.phoneme._get_epitran")
    def test_basic_transliteration(self, mock_get_epitran, mock_load_model):
        """Basic transliteration returns vocab-matched tokens."""
        from app.services.phoneme import _text_to_phonemes

        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "ab"
        mock_get_epitran.return_value = mock_epi

        vocab = {"a": 1, "b": 2}
        mock_load_model.return_value = (None, None, vocab, None, None)

        result = _text_to_phonemes("test", "es")
        assert result == ["a", "b"]

    @patch("app.services.phoneme._load_model")
    @patch("app.services.phoneme._get_epitran")
    def test_chars_not_in_vocab_skipped(self, mock_get_epitran, mock_load_model):
        """Characters not in vocabulary are silently skipped."""
        from app.services.phoneme import _text_to_phonemes

        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "axb"
        mock_get_epitran.return_value = mock_epi

        vocab = {"a": 1, "b": 2}
        mock_load_model.return_value = (None, None, vocab, None, None)

        result = _text_to_phonemes("test", "es")
        assert result == ["a", "b"]


# ===== _gop_sync() =====


class TestGopSync:
    """GOP scoring tests (mock _load_model + _get_log_probs)."""

    @patch("app.services.phoneme._get_log_probs")
    @patch("app.services.phoneme._load_model")
    def test_returns_scored_phonemes(self, mock_load_model, mock_get_log_probs):
        """Returns scored phonemes with correct keys."""
        from app.services.phoneme import _gop_sync

        vocab = {"<pad>": 0, "a": 1, "b": 2}
        id_to_token = {0: "<pad>", 1: "a", 2: "b"}
        blank_id = 0
        mock_load_model.return_value = (None, None, vocab, id_to_token, blank_id)

        T, V = 10, 3
        lp = torch.full((T, V), -10.0)
        lp[0:5, 1] = -0.1
        lp[5:10, 2] = -0.1
        mock_get_log_probs.return_value = (lp, 0.02)

        results = _gop_sync("/fake/audio.wav", ["a", "b"])
        assert len(results) > 0
        for r in results:
            assert "phoneme" in r
            assert "expected" in r
            assert "score" in r
            assert "is_correct" in r
            assert "start" in r
            assert "end" in r

    @patch("app.services.phoneme._get_log_probs")
    @patch("app.services.phoneme._load_model")
    def test_empty_when_no_phonemes_map(self, mock_load_model, mock_get_log_probs):
        """Empty result when no phonemes map to vocab."""
        from app.services.phoneme import _gop_sync

        vocab = {"<pad>": 0, "a": 1}
        id_to_token = {0: "<pad>", 1: "a"}
        blank_id = 0
        mock_load_model.return_value = (None, None, vocab, id_to_token, blank_id)

        lp = torch.randn(10, 2)
        mock_get_log_probs.return_value = (lp, 0.02)

        results = _gop_sync("/fake/audio.wav", ["z", "q"])
        assert results == []
