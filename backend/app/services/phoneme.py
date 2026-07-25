"""Phoneme alignment and GOP scoring using wav2vec2.

Uses facebook/wav2vec2-xlsr-53-espeak-cv-ft for frame-level phoneme
posteriors, CTC forced alignment for mapping expected phonemes to audio
frames, and GOP (Goodness of Pronunciation) scoring.
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import threading
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from app.config import settings

log = logging.getLogger(__name__)

# espeak-ng library path for phonemizer (macOS Homebrew)
if not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
    for candidate in (
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/usr/lib/libespeak-ng.so",
        "/usr/local/lib/libespeak-ng.dylib",
    ):
        if Path(candidate).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = candidate
            break

# Epitran language code mapping
EPITRAN_LANG = {
    "es": "spa-Latn",
    "hr": "hrv-Latn",
    "de": "deu-Latn",
    "zh": "cmn-Hans",
}

# GOP score threshold: above this = acceptable pronunciation
GOP_THRESHOLD = 0.3

# Thread-safe model loading
_model_lock = threading.Lock()
_model_cache: tuple | None = None


def _load_model() -> tuple:
    """Load and cache wav2vec2 phoneme model, processor, and vocab (thread-safe)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    with _model_lock:
        if _model_cache is not None:
            return _model_cache
        log.info("Loading wav2vec2 model: %s", settings.wav2vec2_model)
        processor = Wav2Vec2Processor.from_pretrained(settings.wav2vec2_model)
        model = Wav2Vec2ForCTC.from_pretrained(settings.wav2vec2_model)
        model.eval()

        vocab = processor.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        blank_id = processor.tokenizer.pad_token_id

        log.info("Model loaded. Vocab size: %d, blank_id: %d", len(vocab), blank_id)
        _model_cache = (processor, model, vocab, id_to_token, blank_id)
        return _model_cache


def _ensure_wav_16k(input_path: str) -> str:
    """Ensure audio is 16kHz mono 16-bit WAV. Converts via ffmpeg if needed."""
    path = Path(input_path)
    if path.suffix == ".wav":
        try:
            with wave.open(input_path, "rb") as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2:
                    return input_path
        except Exception:
            pass

    fd, wav_path = tempfile.mkstemp(suffix="_phoneme.wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        Path(wav_path).unlink(missing_ok=True)
        raise RuntimeError(f"Audio conversion failed: {e}") from e
    return wav_path


def _load_wav(path: str) -> np.ndarray:
    """Load 16kHz mono WAV as float32 numpy array in [-1, 1]."""
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _get_log_probs(audio_path: str) -> tuple[torch.Tensor, float]:
    """Run wav2vec2 inference, returning frame-level log probs and frame duration.

    Returns:
        (log_probs, frame_duration) — log_probs is (T, V), frame_duration in seconds.
    """
    processor, model, _, _, _ = _load_model()

    wav_path = _ensure_wav_16k(audio_path)
    try:
        audio = _load_wav(wav_path)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        del audio  # free numpy array
        with torch.no_grad():
            logits = model(**inputs).logits
        del inputs  # free input tensors

        log_probs = torch.log_softmax(logits, dim=-1)[0]  # (T, V)
        del logits  # free logits

        frame_duration = wav_path and 0  # recompute from audio length
        # Recompute from the WAV file directly
        with wave.open(wav_path, "rb") as wf:
            n_samples = wf.getnframes()
        frame_duration = n_samples / 16000 / log_probs.shape[0]

        log.debug("Inference: %d samples -> %d frames", n_samples, log_probs.shape[0])

        return log_probs, frame_duration
    finally:
        if wav_path != audio_path:
            Path(wav_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CTC greedy decode
# ---------------------------------------------------------------------------

def _greedy_decode(
    log_probs: torch.Tensor,
    id_to_token: dict[int, str],
    blank_id: int,
    frame_duration: float,
) -> list[dict]:
    """CTC greedy decoding — recognized phonemes with timestamps and confidence."""
    preds = torch.argmax(log_probs, dim=-1)

    segments: list[dict] = []
    prev_id = blank_id
    start_frame = 0
    frame_scores: list[float] = []

    for t in range(len(preds)):
        tid = preds[t].item()
        if tid != prev_id:
            if prev_id != blank_id and prev_id in id_to_token:
                avg = sum(frame_scores) / len(frame_scores) if frame_scores else -10.0
                segments.append({
                    "phoneme": id_to_token[prev_id],
                    "start": round(start_frame * frame_duration, 3),
                    "end": round(t * frame_duration, 3),
                    "confidence": round(float(np.exp(avg)), 3),
                })
            start_frame = t
            frame_scores = []
            prev_id = tid

        if tid != blank_id:
            frame_scores.append(log_probs[t, tid].item())

    # Emit last segment
    if prev_id != blank_id and prev_id in id_to_token:
        avg = sum(frame_scores) / len(frame_scores) if frame_scores else -10.0
        segments.append({
            "phoneme": id_to_token[prev_id],
            "start": round(start_frame * frame_duration, 3),
            "end": round(len(preds) * frame_duration, 3),
            "confidence": round(float(np.exp(avg)), 3),
        })

    return segments


# ---------------------------------------------------------------------------
# CTC forced alignment (Viterbi)
# ---------------------------------------------------------------------------

def _ctc_forced_align(
    log_probs: torch.Tensor,
    target_ids: list[int],
    blank_id: int,
) -> list[tuple[int, int, int, float]]:
    """CTC forced alignment via Viterbi dynamic programming.

    Aligns a target phoneme sequence to audio frames using the CTC topology:
    ε t₀ ε t₁ ε ... ε tₛ₋₁ ε  (blanks interleaved with targets).

    Returns:
        List of (target_idx, start_frame, end_frame, avg_log_prob).
    """
    T = log_probs.shape[0]
    S = len(target_ids)

    # Expanded CTC target sequence
    L = 2 * S + 1

    if S == 0 or T == 0 or T < L:
        log.warning("Forced alignment skipped: T=%d, S=%d (need T >= %d)", T, S, L)
        return []

    expanded = []
    for s in range(S):
        expanded.append(blank_id)
        expanded.append(target_ids[s])
    expanded.append(blank_id)

    NEG_INF = -1e9

    # Pre-extract emission scores as numpy: (T, L) array
    expanded_ids = np.array(expanded, dtype=np.int64)
    emission = log_probs[:, expanded_ids].numpy()  # (T, L)

    # Viterbi DP
    alpha = np.full((T, L), NEG_INF, dtype=np.float64)
    backptr = np.zeros((T, L), dtype=np.int32)

    alpha[0, 0] = emission[0, 0]
    if L > 1:
        alpha[0, 1] = emission[0, 1]

    for t in range(1, T):
        for l_idx in range(L):
            # Stay in same state
            best = alpha[t - 1, l_idx]
            bp = l_idx

            # Transition from previous state
            if l_idx > 0 and alpha[t - 1, l_idx - 1] > best:
                best = alpha[t - 1, l_idx - 1]
                bp = l_idx - 1

            # Skip blank: non-blank can come from 2 states back if labels differ
            if l_idx > 1 and expanded[l_idx] != blank_id and expanded[l_idx] != expanded[l_idx - 2]:
                if alpha[t - 1, l_idx - 2] > best:
                    best = alpha[t - 1, l_idx - 2]
                    bp = l_idx - 2

            alpha[t, l_idx] = best + emission[t, l_idx]
            backptr[t, l_idx] = bp

    # Backtrace from best final state (last blank or last token)
    if L >= 2 and alpha[T - 1, L - 1] >= alpha[T - 1, L - 2]:
        l_idx = L - 1
    else:
        l_idx = max(L - 2, 0)

    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = l_idx
    for t in range(T - 2, -1, -1):
        l_idx = backptr[t + 1, l_idx]
        path[t] = l_idx

    # Extract per-phoneme segments from the alignment path
    segments: list[tuple[int, int, int, float]] = []
    cur_state = -1
    start = 0
    scores: list[float] = []

    for t in range(T):
        state = int(path[t])
        tok = expanded[state]

        if state != cur_state:
            # Emit previous segment if it was a real phoneme
            if cur_state >= 0 and expanded[cur_state] != blank_id:
                idx = (cur_state - 1) // 2
                avg_lp = sum(scores) / len(scores) if scores else NEG_INF
                segments.append((idx, start, t - 1, avg_lp))
            cur_state = state
            start = t
            scores = []

        if tok != blank_id:
            scores.append(emission[t, state])

    # Emit final segment
    if cur_state >= 0 and expanded[cur_state] != blank_id:
        idx = (cur_state - 1) // 2
        avg_lp = sum(scores) / len(scores) if scores else NEG_INF
        segments.append((idx, start, T - 1, avg_lp))

    return segments


# ---------------------------------------------------------------------------
# Phoneme vocabulary mapping
# ---------------------------------------------------------------------------

def _find_vocab_id(phoneme: str, vocab: dict[str, int]) -> int | None:
    """Map an IPA phoneme to the model's vocabulary ID.

    Tries exact match first, then strips common IPA diacritics.
    """
    if phoneme in vocab:
        return vocab[phoneme]
    # Strip suprasegmentals / diacritics
    stripped = phoneme
    for ch in ("ː", "ˈ", "ˌ", "\u0303", "ʰ", "ʲ", "ʷ", "\u0325", "\u0329"):
        stripped = stripped.replace(ch, "")
    if stripped and stripped in vocab:
        return vocab[stripped]
    return None


@lru_cache(maxsize=4)
def _get_epitran(language: str):
    """Cache epitran instances to avoid repeated initialization."""
    import epitran
    lang_code = EPITRAN_LANG.get(language, "spa-Latn")
    return epitran.Epitran(lang_code)


def _text_to_phonemes(text: str, language: str) -> list[str]:
    """Convert text to a list of phoneme tokens matching the model vocabulary.

    Uses epitran for grapheme-to-phoneme, then splits the IPA string into
    tokens that exist in the wav2vec2 vocabulary (greedy longest-match).
    """
    epi = _get_epitran(language)
    ipa_str = epi.transliterate(text)

    _, _, vocab, _, _ = _load_model()

    # Greedy longest-match tokenization against model vocab
    tokens: list[str] = []
    i = 0
    while i < len(ipa_str):
        best_match = None
        # Try longest match first (up to 6 chars covers all vocab entries)
        for length in range(min(6, len(ipa_str) - i), 0, -1):
            candidate = ipa_str[i : i + length]
            if candidate in vocab:
                best_match = candidate
                break
        if best_match:
            tokens.append(best_match)
            i += len(best_match)
        else:
            # Skip characters not in vocab (spaces, stress marks, etc.)
            i += 1

    return tokens


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

def _align_sync(audio_path: str) -> list[dict]:
    """Greedy CTC decode (synchronous)."""
    log_probs, fd = _get_log_probs(audio_path)
    _, _, _, id2tok, blank = _load_model()
    return _greedy_decode(log_probs, id2tok, blank, fd)


def _gop_sync(audio_path: str, expected_phonemes: list[str]) -> list[dict]:
    """Forced alignment + GOP scoring (synchronous)."""
    _, _, vocab, id2tok, blank = _load_model()
    log_probs, fd = _get_log_probs(audio_path)

    # Map phonemes to vocab IDs, skipping unmappable ones
    targets: list[tuple[int, str]] = []
    for p in expected_phonemes:
        vid = _find_vocab_id(p, vocab)
        if vid is not None and vid != blank:
            targets.append((vid, p))

    if not targets:
        log.warning("No phonemes could be mapped to model vocabulary")
        return []

    target_ids = [t[0] for t in targets]
    target_phones = [t[1] for t in targets]

    alignment = _ctc_forced_align(log_probs, target_ids, blank)

    results: list[dict] = []
    for tidx, start, end, avg_lp in alignment:
        if tidx >= len(target_phones):
            continue

        gop = float(np.exp(avg_lp))

        # Identify what the model actually hears in this segment
        seg_probs = log_probs[start : end + 1]
        if len(seg_probs) > 0:
            avg_seg = seg_probs.mean(dim=0)
            best_id = torch.argmax(avg_seg).item()
            actual = id2tok.get(best_id, "?")
            if actual in ("<pad>", "<s>", "</s>", "<unk>"):
                actual = "∅"  # silence / no speech detected
        else:
            actual = "?"

        results.append({
            "phoneme": actual,
            "expected": target_phones[tidx],
            "score": round(gop, 3),
            "is_correct": gop > GOP_THRESHOLD,
            "start": round(start * fd, 3),
            "end": round((end + 1) * fd, 3),
        })

    return results


async def align_phonemes(audio_path: str, transcript: str = "", language: str = "es") -> list[dict]:
    """Run CTC greedy decode on audio — what phonemes does the model hear?

    Returns list of dicts with phoneme, start, end, confidence.

    Note: transcript and language are accepted for API compatibility but not
    used in greedy decoding.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _align_sync, audio_path)


async def compute_gop_scores(
    audio_path: str,
    expected_phonemes: list[str],
    aligned_phonemes: list[dict] | None = None,
) -> list[dict]:
    """Compute GOP scores by force-aligning expected phonemes against audio.

    Args:
        audio_path: Path to audio file (any format — converted to WAV internally).
        expected_phonemes: Phoneme tokens in the model's vocabulary format.
            Use text_to_phonemes() to convert from text.
        aligned_phonemes: Deprecated — ignored. Forced alignment is computed internally.

    Returns:
        List of dicts with phoneme (actual), expected, score, is_correct, start, end.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _gop_sync, audio_path, expected_phonemes)


async def text_to_phonemes(text: str, language: str = "es") -> list[str]:
    """Convert text to phoneme tokens matching the model vocabulary.

    Convenience wrapper for the speech analysis pipeline:
    text → epitran IPA → model vocab tokens.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _text_to_phonemes, text, language)
