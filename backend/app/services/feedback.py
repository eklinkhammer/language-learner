"""Human-readable pronunciation feedback using epitran and panphon.

Compares expected vs. actual phonemes using articulatory features to produce
specific, actionable feedback (e.g., "Try voicing this sound" or "Move your
tongue further forward").
"""

import asyncio
import logging
from functools import lru_cache

import epitran
import panphon
import panphon.distance

from app.data.phoneme_examples import get_example
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
        tts_url=f"/api/tts?text={ex.word}&language=en",
    )


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

    for p in incorrect:
        expected_ph = p.get("expected", "")
        actual_ph = p.get("phoneme", "")

        pair = (expected_ph, actual_ph)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        # Collect silence/unknown — summarize once at the end
        if actual_ph in ("∅", "?", ""):
            silence_phonemes.append(expected_ph)
            continue

        # Build feedback line
        line = f"  /{expected_ph}/ → you said /{actual_ph}/"

        # Get articulatory tip (uses panphon features internally)
        tip = _describe_difference(expected_ph, actual_ph)
        if tip is None:
            line += " — very close, minor adjustment needed."
            tip_text = "Very close, minor adjustment needed."
        else:
            line += f" — {tip}"
            tip_text = tip

        feedback.append(line)
        items.append(FeedbackItem(
            type="phoneme_error",
            message=f"/{expected_ph}/ → you said /{actual_ph}/ — {tip_text}",
            expected_phoneme=expected_ph,
            actual_phoneme=actual_ph,
            expected_example=_make_example_ref(expected_ph),
            actual_example=_make_example_ref(actual_ph),
        ))

    # Summarize undetected phonemes
    if silence_phonemes:
        joined = "/, /".join(silence_phonemes)
        msg = f"/{joined}/ — not detected. Try speaking more clearly or closer to the mic."
        feedback.append(f"  {msg}")
        items.append(FeedbackItem(type="silence_error", message=msg))

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
