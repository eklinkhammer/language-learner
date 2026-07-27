"""Tests for app.services.feedback — pronunciation feedback generation."""

from unittest.mock import MagicMock, patch
from urllib.parse import quote

import pytest


# ===== _describe_difference() =====


class TestDescribeDifference:
    """Articulatory difference description tests."""

    @patch("app.services.feedback._get_panphon")
    def test_common_tips_exact_match(self, mock_panphon):
        """COMMON_TIPS exact match ('s' → 'θ')."""
        from app.services.feedback import _describe_difference

        result = _describe_difference("s", "θ")
        assert result is not None
        assert "th" in result.lower() or "think" in result.lower()

    @patch("app.services.feedback._get_panphon")
    def test_articulatory_voicing_difference(self, mock_panphon):
        """Articulatory voicing difference ('d' vs 't') detected via features."""
        from app.services.feedback import _describe_difference

        # Mock panphon to return feature vectors that differ in voicing
        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_panphon.return_value = (mock_ft, mock_dist)

        # Create mock feature segments with voicing difference
        fts_d = MagicMock()
        fts_t = MagicMock()
        # "d" is voiced (+1), "t" is voiceless (-1)
        fts_d.numeric.return_value = [1, -1, -1, -1, -1]
        fts_t.numeric.return_value = [-1, -1, -1, -1, -1]
        mock_ft.fts.side_effect = lambda p: fts_d if p == "d" else fts_t
        mock_ft.names = ["voi", "nas", "cont", "strid", "lat"]

        result = _describe_difference("d", "t")
        assert result is not None
        assert "voiced" in result.lower()

    @patch("app.services.feedback._get_panphon")
    def test_unknown_phoneme_returns_none(self, mock_panphon):
        """Unknown phoneme → None."""
        from app.services.feedback import _describe_difference

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_panphon.return_value = (mock_ft, mock_dist)
        mock_ft.fts.return_value = None

        result = _describe_difference("ʘ", "ǂ")
        assert result is None

    @patch("app.services.feedback._get_panphon")
    def test_identical_features_returns_none(self, mock_panphon):
        """Identical features → None."""
        from app.services.feedback import _describe_difference

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_panphon.return_value = (mock_ft, mock_dist)

        fts_mock = MagicMock()
        fts_mock.numeric.return_value = [1, -1, 1]
        mock_ft.fts.return_value = fts_mock
        mock_ft.names = ["voi", "nas", "cont"]

        result = _describe_difference("a", "a")
        assert result is None

    @patch("app.services.feedback._get_panphon")
    def test_unspecified_features_skipped(self, mock_panphon):
        """Unspecified (0) features are skipped in diff."""
        from app.services.feedback import _describe_difference

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_panphon.return_value = (mock_ft, mock_dist)

        fts_exp = MagicMock()
        fts_act = MagicMock()
        # Feature differs but one side is 0 (unspecified) → skip
        fts_exp.numeric.return_value = [0, 1]
        fts_act.numeric.return_value = [1, 1]
        mock_ft.fts.side_effect = lambda p: fts_exp if p == "x" else fts_act
        mock_ft.names = ["voi", "nas"]

        result = _describe_difference("x", "y")
        assert result is None

    @patch("app.services.feedback._get_panphon")
    def test_priority_ordering(self, mock_panphon):
        """Voicing difference is reported before place difference."""
        from app.services.feedback import _describe_difference

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_panphon.return_value = (mock_ft, mock_dist)

        fts_exp = MagicMock()
        fts_act = MagicMock()
        # Both voicing and place differ
        fts_exp.numeric.return_value = [1, 1]
        fts_act.numeric.return_value = [-1, -1]
        mock_ft.fts.side_effect = lambda p: fts_exp if p == "x" else fts_act
        mock_ft.names = ["ant", "voi"]  # "ant" first in vector but "voi" has priority

        result = _describe_difference("x", "y")
        assert result is not None
        assert "voiced" in result.lower()


# ===== _generate_feedback_sync() =====


class TestGenerateFeedbackSync:
    """Feedback generation tests (mock epitran + panphon)."""

    def _get_patched_func(self):
        """Return _generate_feedback_sync with mocked epitran/panphon."""
        from app.services.feedback import _generate_feedback_sync
        return _generate_feedback_sync

    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_all_correct(self, mock_panphon, mock_epitran, sample_gop_scores_all_correct):
        """All correct → 'Excellent pronunciation!'."""
        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "abc"
        mock_epitran.return_value = mock_epi

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_dist.feature_edit_distance_div_maxlen.return_value = 0.0
        mock_panphon.return_value = (mock_ft, mock_dist)

        func = self._get_patched_func()
        result, items = func("test", "test", sample_gop_scores_all_correct, "es")
        assert any("Excellent" in line for line in result)

    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_empty_scores(self, mock_panphon, mock_epitran):
        """Empty scores → 'No phonemes could be analyzed'."""
        mock_epi = MagicMock()
        mock_epitran.return_value = mock_epi

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_panphon.return_value = (mock_ft, mock_dist)

        func = self._get_patched_func()
        result, items = func("test", "test", [], "es")
        assert any("No phonemes" in line for line in result)

    @patch("app.services.feedback._describe_difference")
    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_80_percent_bucket(self, mock_panphon, mock_epitran, mock_describe):
        """80% correct bucket message."""
        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "test"
        mock_epitran.return_value = mock_epi

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_dist.feature_edit_distance_div_maxlen.return_value = 0.1
        mock_panphon.return_value = (mock_ft, mock_dist)
        mock_describe.return_value = "Try something."

        # 4 correct, 1 incorrect = 80%
        scores = [
            {"phoneme": "a", "expected": "a", "score": 0.9, "is_correct": True},
            {"phoneme": "b", "expected": "b", "score": 0.9, "is_correct": True},
            {"phoneme": "c", "expected": "c", "score": 0.9, "is_correct": True},
            {"phoneme": "d", "expected": "d", "score": 0.9, "is_correct": True},
            {"phoneme": "t", "expected": "s", "score": 0.1, "is_correct": False},
        ]

        func = self._get_patched_func()
        result, items = func("test", "test", scores, "es")
        assert any("Good" in line for line in result)

    @patch("app.services.feedback._describe_difference")
    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_50_percent_bucket(self, mock_panphon, mock_epitran, mock_describe):
        """50% correct bucket message."""
        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "test"
        mock_epitran.return_value = mock_epi

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_dist.feature_edit_distance_div_maxlen.return_value = 0.3
        mock_panphon.return_value = (mock_ft, mock_dist)
        mock_describe.return_value = "Try something."

        scores = [
            {"phoneme": "a", "expected": "a", "score": 0.9, "is_correct": True},
            {"phoneme": "t", "expected": "s", "score": 0.1, "is_correct": False},
        ]

        func = self._get_patched_func()
        result, items = func("test", "test", scores, "es")
        assert any("Decent" in line for line in result)

    @patch("app.services.feedback._describe_difference")
    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_below_50_percent_bucket(self, mock_panphon, mock_epitran, mock_describe):
        """Below 50% correct bucket message."""
        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "test"
        mock_epitran.return_value = mock_epi

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_dist.feature_edit_distance_div_maxlen.return_value = 0.7
        mock_panphon.return_value = (mock_ft, mock_dist)
        mock_describe.return_value = "Try something."

        scores = [
            {"phoneme": "x", "expected": "a", "score": 0.05, "is_correct": False},
            {"phoneme": "y", "expected": "b", "score": 0.05, "is_correct": False},
            {"phoneme": "z", "expected": "c", "score": 0.05, "is_correct": False},
        ]

        func = self._get_patched_func()
        result, items = func("test", "test", scores, "es")
        assert any("Keep practicing" in line for line in result)

    @patch("app.services.feedback._describe_difference")
    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_includes_articulatory_tips(self, mock_panphon, mock_epitran, mock_describe):
        """Feedback includes articulatory tips for incorrect phonemes."""
        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "test"
        mock_epitran.return_value = mock_epi

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_dist.feature_edit_distance_div_maxlen.return_value = 0.1
        mock_panphon.return_value = (mock_ft, mock_dist)
        mock_describe.return_value = "Try voicing it."

        scores = [
            {"phoneme": "a", "expected": "a", "score": 0.9, "is_correct": True},
            {"phoneme": "t", "expected": "d", "score": 0.1, "is_correct": False},
        ]

        func = self._get_patched_func()
        result, items = func("test", "test", scores, "es")
        assert any("Try voicing it." in line for line in result)

    @patch("app.services.feedback._describe_difference")
    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_deduplicates_same_error_pair(self, mock_panphon, mock_epitran, mock_describe):
        """Same error pair (expected, actual) only reported once."""
        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "test"
        mock_epitran.return_value = mock_epi

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_dist.feature_edit_distance_div_maxlen.return_value = 0.1
        mock_panphon.return_value = (mock_ft, mock_dist)
        mock_describe.return_value = "Tip."

        scores = [
            {"phoneme": "t", "expected": "d", "score": 0.1, "is_correct": False},
            {"phoneme": "t", "expected": "d", "score": 0.1, "is_correct": False},
            {"phoneme": "a", "expected": "a", "score": 0.9, "is_correct": True},
        ]

        func = self._get_patched_func()
        result, items = func("test", "test", scores, "es")
        # Only one line should mention /d/ → /t/
        tip_lines = [l for l in result if "/d/" in l and "/t/" in l]
        assert len(tip_lines) == 1

    @patch("app.services.feedback._describe_difference")
    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_silence_phonemes_summarized(self, mock_panphon, mock_epitran, mock_describe,
                                         sample_gop_scores_with_silence):
        """Silence phonemes are summarized together."""
        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "test"
        mock_epitran.return_value = mock_epi

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_dist.feature_edit_distance_div_maxlen.return_value = 0.3
        mock_panphon.return_value = (mock_ft, mock_dist)
        mock_describe.return_value = None

        func = self._get_patched_func()
        result, items = func("test", "test", sample_gop_scores_with_silence, "es")
        # Should have a line summarizing undetected phonemes
        silence_lines = [l for l in result if "not detected" in l]
        assert len(silence_lines) == 1
        assert "/b/" in silence_lines[0] and "/c/" in silence_lines[0]

    @patch("app.services.feedback._describe_difference")
    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_example_hint_present_for_known_phonemes(
        self, mock_panphon, mock_epitran, mock_describe
    ):
        """When both expected and actual phonemes have examples, hint is included."""
        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "test"
        mock_epi.word_to_tuples.return_value = []
        mock_epitran.return_value = mock_epi

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_dist.feature_edit_distance_div_maxlen.return_value = 0.1
        mock_panphon.return_value = (mock_ft, mock_dist)
        mock_describe.return_value = "Tip."

        # "d" and "t" both have English examples in PHONEME_EXAMPLES
        scores = [
            {"phoneme": "t", "expected": "d", "score": 0.1, "is_correct": False},
            {"phoneme": "a", "expected": "a", "score": 0.9, "is_correct": True},
        ]

        func = self._get_patched_func()
        result, items = func("test", "test", scores, "es")

        # Check feedback lines contain the "as in" example hints
        error_lines = [l for l in result if "you said" in l and "as in" in l]
        assert len(error_lines) == 1
        assert 'you said t as in "top"' in error_lines[0]
        assert 'you want d as in "dog"' in error_lines[0]

        # Check structured FeedbackItem message
        error_items = [it for it in items if it.type == "phoneme_error"]
        assert len(error_items) == 1
        assert 'you said t as in "top"' in error_items[0].message
        assert 'you want d as in "dog"' in error_items[0].message

    @patch("app.services.feedback._describe_difference")
    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_example_hint_absent_for_unknown_phonemes(
        self, mock_panphon, mock_epitran, mock_describe
    ):
        """When neither phoneme has an example, the parenthetical hint is absent."""
        mock_epi = MagicMock()
        mock_epi.transliterate.return_value = "test"
        mock_epi.word_to_tuples.return_value = []
        mock_epitran.return_value = mock_epi

        mock_ft = MagicMock()
        mock_dist = MagicMock()
        mock_dist.feature_edit_distance_div_maxlen.return_value = 0.1
        mock_panphon.return_value = (mock_ft, mock_dist)
        mock_describe.return_value = "Tip."

        # "ɾ" and "β" both map to None in PHONEME_EXAMPLES
        scores = [
            {"phoneme": "β", "expected": "ɾ", "score": 0.1, "is_correct": False},
            {"phoneme": "a", "expected": "a", "score": 0.9, "is_correct": True},
        ]

        func = self._get_patched_func()
        result, items = func("test", "test", scores, "es")

        # No "as in" hint should appear
        error_lines = [l for l in result if "/ɾ/" in l]
        assert len(error_lines) == 1
        assert "as in" not in error_lines[0]

        error_items = [it for it in items if it.type == "phoneme_error"]
        assert len(error_items) == 1
        assert "as in" not in error_items[0].message


# ===== _make_example_ref() =====


class TestMakeExampleRef:
    """Tests for _make_example_ref helper."""

    def test_known_phoneme_returns_ref(self):
        """Known phoneme (e.g. 'd') returns a PhonemeExampleRef with correct fields."""
        from app.services.feedback import _make_example_ref

        ref = _make_example_ref("d")
        assert ref is not None
        assert ref.phoneme == "d"
        assert ref.example_word == "dog"
        assert ref.highlight == "d"
        assert ref.description == 'like the "d" in "dog"'
        assert ref.tts_url == f"/api/tts?text={quote('dog')}&language=en"

    def test_unknown_phoneme_returns_none(self):
        """Unknown phoneme (e.g. 'ɾ') returns None."""
        from app.services.feedback import _make_example_ref

        ref = _make_example_ref("ɾ")
        assert ref is None


# ===== _build_word_breakdown() =====


class TestBuildWordBreakdown:
    """Tests for _build_word_breakdown helper."""

    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_normal_word_produces_breakdown(self, mock_panphon, mock_epitran):
        """Normal word produces expected breakdown string."""
        from app.services.feedback import _build_word_breakdown

        mock_epi = MagicMock()
        # Simulate word_to_tuples for "bat" → b, a, t
        mock_epi.word_to_tuples.return_value = [
            ("L", "L", "b", "b", []),
            ("L", "L", "a", "a", []),
            ("L", "L", "t", "t", []),
        ]

        result = _build_word_breakdown("bat", mock_epi)

        assert 'b as in "bat"' in result
        assert 'a as in "father"' in result
        assert 't as in "top"' in result
        assert result.endswith('for "bat"')

    @patch("app.services.feedback._get_epitran")
    @patch("app.services.feedback._get_panphon")
    def test_empty_ipa_entries_skipped(self, mock_panphon, mock_epitran):
        """Entries with empty ipa_str (combining accents) are skipped."""
        from app.services.feedback import _build_word_breakdown

        mock_epi = MagicMock()
        # Second entry has empty ipa_str (e.g. combining accent)
        mock_epi.word_to_tuples.return_value = [
            ("L", "L", "b", "b", []),
            ("L", "L", "\u0301", "", []),   # combining acute accent, empty IPA
            ("L", "L", "a", "a", []),
        ]

        result = _build_word_breakdown("ba", mock_epi)

        # Should only have two parts (b and a), accent is skipped
        assert 'b as in "bat"' in result
        assert 'a as in "father"' in result
        assert "\u0301" not in result
        assert result.endswith('for "ba"')
