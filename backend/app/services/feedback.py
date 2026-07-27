"""Human-readable pronunciation feedback using epitran and panphon.

Compares expected vs. actual phonemes using articulatory features to produce
specific, actionable feedback (e.g., "Try voicing this sound" or "Move your
tongue further forward").
"""

import asyncio
import logging
from functools import lru_cache
from urllib.parse import quote

import epitran
import panphon
import panphon.distance

from app.data.phoneme_examples import PHONEME_EXAMPLES, get_example
from app.models.schemas import FeedbackItem, PhonemeExampleRef
from app.services.phoneme import EPITRAN_LANG

log = logging.getLogger(__name__)

# Human-readable names for articulatory features
FEATURE_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    # feature name → (description when +, description when -)
    "voi":    ("voiced (vocal cords vibrate)", "voiceless (no vibration)"),
    "nas":    ("nasal (air through nose)", "oral (air through mouth)"),
    "cont":   ("continuant (air flows freely)", "stop (air blocked briefly)"),
    "strid":  ("strident (high-energy friction)", "non-strident"),
    "lat":    ("lateral (air around tongue sides)", "central"),
    "ant":    ("anterior (front of mouth)", "posterior (back of mouth)"),
    "cor":    ("coronal (tongue tip/blade)", "non-coronal"),
    "distr":  ("distributed (broad contact)", "focused (narrow contact)"),
    "lab":    ("labial (lips involved)", "non-labial"),
    "hi":     ("high tongue position", "non-high tongue"),
    "lo":     ("low tongue position", "non-low tongue"),
    "back":   ("back tongue position", "front tongue position"),
    "round":  ("rounded lips", "unrounded lips"),
    "syl":    ("syllabic", "non-syllabic"),
    "son":    ("sonorant", "obstruent"),
    "cons":   ("consonantal", "non-consonantal"),
    "sg":     ("spread glottis (aspirated)", "non-aspirated"),
    "cg":     ("constricted glottis", "non-constricted"),
    "delrel": ("delayed release (affricate)", "non-affricate"),
    "long":   ("long", "short"),
    "tense":  ("tense", "lax"),
}

# Common substitution patterns → plain-English tips
COMMON_TIPS: dict[tuple[str, str], str] = {
    ("s", "θ"):  "In many Spanish dialects, this is pronounced like English 'th' in 'think'.",
    ("θ", "s"):  "This region uses 's' instead of 'th' — both are accepted.",
    ("b", "β"):  "Between vowels, Spanish 'b' softens — keep your lips slightly apart.",
    ("d", "ð"):  "Between vowels, Spanish 'd' softens — like 'th' in English 'the'.",
    ("ɡ", "ɣ"):  "Between vowels, Spanish 'g' softens — don't fully close the back of your mouth.",
    ("ɾ", "r"):  "Use a single tongue tap, not a full trill.",
    ("r", "ɾ"):  "Use a full tongue trill here, not just a tap.",
    ("r", "ɹ"):  "Roll/trill your tongue — don't use the English 'r'.",
    ("ɾ", "ɹ"):  "Tap your tongue quickly against the ridge — don't use the English 'r'.",
    ("n", "ŋ"):  "Keep your tongue at the front — this should be a dental/alveolar 'n'.",
    ("ŋ", "n"):  "This 'n' is produced at the back of the mouth, before 'g' or 'k'.",
}


@lru_cache(maxsize=4)
def _get_epitran(language: str) -> epitran.Epitran:
    lang_code = EPITRAN_LANG.get(language)
    if lang_code is None:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported: {', '.join(EPITRAN_LANG)}"
        )
    return epitran.Epitran(lang_code)


@lru_cache(maxsize=1)
def _get_panphon() -> tuple[panphon.FeatureTable, panphon.distance.Distance]:
    return panphon.FeatureTable(), panphon.distance.Distance()


def _describe_difference(expected: str, actual: str) -> str | None:
    """Describe the articulatory difference between two phonemes.

    Returns a human-readable hint, or None if no useful description is possible.
    """
    # Check common tips first
    tip = COMMON_TIPS.get((expected, actual))
    if tip:
        return tip

    ft, _ = _get_panphon()
    fts_exp = ft.fts(expected)
    fts_act = ft.fts(actual)

    if not fts_exp or not fts_act:
        return None

    vec_exp = fts_exp.numeric()
    vec_act = fts_act.numeric()
    names = ft.names

    diffs = []
    for i, name in enumerate(names):
        if vec_exp[i] != vec_act[i] and name in FEATURE_DESCRIPTIONS:
            # Skip differences involving unspecified (0) features
            if vec_exp[i] == 0 or vec_act[i] == 0:
                continue
            # What the expected phoneme requires
            pos_desc, neg_desc = FEATURE_DESCRIPTIONS[name]
            needed = pos_desc if vec_exp[i] == 1 else neg_desc
            diffs.append((name, needed))

    if not diffs:
        return None

    # Pick the most salient difference for the tip
    # Priority: voicing > place > manner
    priority = ["voi", "ant", "cor", "hi", "lo", "back", "lab", "round",
                 "nas", "cont", "strid", "lat", "distr"]
    for feat in priority:
        for name, desc in diffs:
            if name == feat:
                return f"Try making it {desc}."

    # Fallback to first difference
    return f"Try making it {diffs[0][1]}."


def _make_example_ref(phoneme: str) -> PhonemeExampleRef | None:
    """Build a PhonemeExampleRef with TTS URL, or None if no English example."""
    ex = get_example(phoneme)
    if ex is None:
        return None
    return PhonemeExampleRef(
        phoneme=phoneme,
        example_word=ex.word,
        highlight=ex.highlight,
        description=ex.description,
        tts_url=f"/api/tts?text={quote(ex.word)}&language=en",
    )


def _map_phonemes_to_words(
    expected_text: str,
    phoneme_scores: list[dict],
    epi: epitran.Epitran,
) -> dict[int, tuple[str, str]]:
    """Map each phoneme_score index to (source_word, grapheme).

    Uses epitran's word_to_tuples to get grapheme→phoneme mappings, then walks
    phoneme_scores in order to match each expected phoneme to its source word
    and the letter(s) that produce it.

    Returns:
        {score_index: (word, grapheme)} mapping.
    """
    words = expected_text.split()
    if not words:
        return {}

    # Build flat list of (phoneme_char, word, grapheme) from word_to_tuples
    ipa_entries: list[tuple[str, str, str]] = []
    for word in words:
        tuples = epi.word_to_tuples(word)
        for tup in tuples:
            # tup format: (category, ?, grapheme, ipa_string, features_list)
            grapheme = tup[2]
            ipa_str = tup[3]
            # Each char in ipa_str maps back to this grapheme
            for ch in ipa_str:
                ipa_entries.append((ch, word, grapheme))

    # Walk phoneme_scores in order, matching each expected phoneme
    result: dict[int, tuple[str, str]] = {}
    ipa_pos = 0
    for score_idx, score in enumerate(phoneme_scores):
        expected_ph = score.get("expected", "")
        if not expected_ph:
            continue
        # Search forward for a matching phoneme char
        search_pos = ipa_pos
        while search_pos < len(ipa_entries):
            if ipa_entries[search_pos][0] == expected_ph:
                result[score_idx] = (ipa_entries[search_pos][1], ipa_entries[search_pos][2])
                ipa_pos = search_pos + 1
                break
            search_pos += 1

    return result


def _build_word_breakdown(word: str, epi: epitran.Epitran) -> str:
    """Build a phonetic breakdown sentence for a word.

    Uses grapheme-phoneme mapping to show letters and their sounds.
    E.g. for "fish": 'f as in "fall", i as in "dish", sh as in "shy" for "fish"'
    """
    tuples = epi.word_to_tuples(word)
    parts: list[str] = []

    for tup in tuples:
        grapheme = tup[2]
        ipa_str = tup[3]

        # Skip entries with no phonemic content (e.g. combining accents)
        if not ipa_str:
            continue

        # Try matching the full IPA string as a multi-char phoneme (e.g. tʃ)
        if ipa_str in PHONEME_EXAMPLES:
            ex = PHONEME_EXAMPLES[ipa_str]
            if ex is not None:
                parts.append(f'{grapheme} as in "{ex.word}"')
            else:
                parts.append(grapheme)
        elif len(ipa_str) == 1:
            # Single phoneme
            ex = get_example(ipa_str)
            if ex is not None:
                parts.append(f'{grapheme} as in "{ex.word}"')
            else:
                parts.append(grapheme)
        else:
            # Multi-phoneme grapheme (e.g. "ue" → "we"): list each phoneme
            sub_parts = []
            for ch in ipa_str:
                ex = get_example(ch)
                if ex is not None:
                    sub_parts.append(f'{ch} as in "{ex.word}"')
                else:
                    sub_parts.append(ch)
            parts.append(f'{grapheme} ({", ".join(sub_parts)})')

    return ", ".join(parts) + f' for "{word}"'


def _generate_feedback_sync(
    expected_text: str,
    transcript: str,
    phoneme_scores: list[dict],
    language: str,
) -> tuple[list[str], list[FeedbackItem]]:
    """Generate feedback (synchronous)."""
    ft_table, dist = _get_panphon()
    epi = _get_epitran(language)

    feedback: list[str] = []
    items: list[FeedbackItem] = []

    # Map each phoneme score to its source word
    phoneme_word_map = _map_phonemes_to_words(expected_text, phoneme_scores, epi)

    if not phoneme_scores:
        msg = "No phonemes could be analyzed from the recording."
        feedback.append(msg)
        items.append(FeedbackItem(type="summary", message=msg))
        return feedback, items

    incorrect = [p for p in phoneme_scores if not p.get("is_correct", True)]
    correct_count = len(phoneme_scores) - len(incorrect)

    # Overall summary
    pct = correct_count / len(phoneme_scores) * 100
    if pct == 100:
        msg = "Excellent pronunciation! All phonemes matched."
        feedback.append(msg)
        items.append(FeedbackItem(type="summary", message=msg))
        return feedback, items
    elif pct >= 80:
        msg = f"Good pronunciation ({pct:.0f}% of phonemes correct). A few areas to refine:"
    elif pct >= 50:
        msg = f"Decent attempt ({pct:.0f}% correct). Focus on these sounds:"
    else:
        msg = f"Keep practicing ({pct:.0f}% correct). Here are the key areas:"

    feedback.append(msg)
    items.append(FeedbackItem(type="summary", message=msg))

    # Group similar errors to avoid repetitive feedback
    seen_pairs: set[tuple[str, str]] = set()
    silence_phonemes: list[str] = []
    # Track which words have errors (for word_breakdown generation)
    error_words: set[str] = set()

    # Build index→score mapping for incorrect phonemes
    incorrect_indices = [
        i for i, p in enumerate(phoneme_scores) if not p.get("is_correct", True)
    ]

    for idx in incorrect_indices:
        p = phoneme_scores[idx]
        expected_ph = p.get("expected", "")
        actual_ph = p.get("phoneme", "")
        word_and_letter = phoneme_word_map.get(idx)
        source_word = word_and_letter[0] if word_and_letter else None
        source_letter = word_and_letter[1] if word_and_letter else None

        pair = (expected_ph, actual_ph)
        if pair in seen_pairs:
            if source_word:
                error_words.add(source_word)
            continue
        seen_pairs.add(pair)

        # Collect silence/unknown — summarize once at the end
        if actual_ph in ("∅", "?", ""):
            silence_phonemes.append(expected_ph)
            continue

        if source_word:
            error_words.add(source_word)

        # Build feedback line
        if source_word and source_letter:
            line = f'  In "{source_word}" (letter "{source_letter}"): /{expected_ph}/ → you said /{actual_ph}/'
        elif source_word:
            line = f'  In "{source_word}": /{expected_ph}/ → you said /{actual_ph}/'
        else:
            line = f"  /{expected_ph}/ → you said /{actual_ph}/"

        # Get articulatory tip (uses panphon features internally)
        tip = _describe_difference(expected_ph, actual_ph)
        if tip is None:
            tip_text = "Very close, minor adjustment needed."
        else:
            tip_text = tip

        # Build "you said X as in ..., you want Y as in ..." explanation
        expected_ref = _make_example_ref(expected_ph)
        actual_ref = _make_example_ref(actual_ph)
        example_parts: list[str] = []
        if actual_ref:
            example_parts.append(
                f'you said {actual_ph} as in "{actual_ref.example_word}"'
            )
        if expected_ref:
            example_parts.append(
                f'you want {expected_ph} as in "{expected_ref.example_word}"'
            )
        example_hint = f" ({', '.join(example_parts)})" if example_parts else ""

        line += f" — {tip_text}{example_hint}"

        feedback.append(line)
        items.append(FeedbackItem(
            type="phoneme_error",
            message=f"/{expected_ph}/ → you said /{actual_ph}/ — {tip_text}{example_hint}",
            expected_phoneme=expected_ph,
            actual_phoneme=actual_ph,
            expected_example=expected_ref,
            actual_example=actual_ref,
            source_word=source_word,
            source_letter=source_letter,
        ))

    # Summarize undetected phonemes
    if silence_phonemes:
        joined = "/, /".join(silence_phonemes)
        msg = f"/{joined}/ — not detected. Try speaking more clearly or closer to the mic."
        feedback.append(f"  {msg}")
        items.append(FeedbackItem(type="silence_error", message=msg))

    # Word breakdown for each word that had errors
    for word in expected_text.split():
        if word in error_words:
            breakdown = _build_word_breakdown(word, epi)
            items.append(FeedbackItem(
                type="word_breakdown",
                message=breakdown,
                source_word=word,
            ))

    # General tips based on overall distance (normalized)
    expected_ipa = epi.transliterate(expected_text)
    transcript_ipa = epi.transliterate(transcript)

    overall_dist = dist.feature_edit_distance_div_maxlen(expected_ipa, transcript_ipa)
    if overall_dist > 0.5:
        feedback.append("")
        tip_msg = (
            "Tip: Try listening to a native speaker say this phrase and repeat slowly, "
            "syllable by syllable."
        )
        feedback.append(tip_msg)
        items.append(FeedbackItem(type="tip", message=tip_msg))

    return feedback, items


async def generate_feedback(
    expected_text: str,
    transcript: str,
    phoneme_scores: list[dict],
    language: str = "es",
) -> tuple[list[str], list[FeedbackItem]]:
    """Generate human-readable pronunciation feedback.

    Uses epitran for grapheme-to-phoneme conversion and panphon for
    articulatory feature-based phoneme distance computation. Produces
    specific tips like "try voicing this sound" based on which articulatory
    features differ between expected and actual pronunciation.

    Args:
        expected_text: The target text the user was trying to say.
        transcript: What Whisper actually transcribed.
        phoneme_scores: GOP results from the phoneme service — list of dicts
            with keys: phoneme, expected, score, is_correct.
        language: Language code (es, hr, de, zh).

    Returns:
        Tuple of (feedback strings, structured FeedbackItem list).
    """
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None, _generate_feedback_sync, expected_text, transcript, phoneme_scores, language
        )
    except Exception:
        log.exception("Feedback generation failed")
        return ["(Pronunciation feedback is temporarily unavailable.)"], []
