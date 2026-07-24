"""Human-readable pronunciation feedback using epitran and panphon."""


async def generate_feedback(
    expected_text: str,
    transcript: str,
    phoneme_scores: list[dict],
    language: str = "es",
) -> list[str]:
    """Generate human-readable pronunciation feedback.

    Uses epitran for grapheme-to-phoneme conversion and panphon
    for articulatory feature-based phoneme distance computation.

    Args:
        expected_text: The target text the user was trying to say.
        transcript: What was actually transcribed.
        phoneme_scores: GOP scores from phoneme service.
        language: Language code.

    Returns:
        List of feedback strings describing pronunciation issues.
    """
    # TODO: Implement with epitran + panphon
    # import epitran
    # import panphon
    # epi = epitran.Epitran(lang_code)  # e.g., "spa-Latn"
    # ft = panphon.FeatureTable()
    # expected_ipa = epi.transliterate(expected_text)
    # actual_ipa = epi.transliterate(transcript)
    # distance = ft.feature_edit_distance(expected_ipa, actual_ipa)
    # ... generate feedback based on which features differ
    return ["(stub) Pronunciation feedback will appear here."]
