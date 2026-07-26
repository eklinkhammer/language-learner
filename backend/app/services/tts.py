import hashlib
import tempfile
from pathlib import Path

import edge_tts

VOICE_MAP: dict[str, str] = {
    "es": "es-ES-AlvaroNeural",
    "hr": "hr-HR-GabrijelaNeural",
    "de": "de-DE-ConradNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "tts_cache"


async def synthesize(text: str, language: str) -> Path:
    """Synthesize text to speech and return the cached MP3 path."""
    CACHE_DIR.mkdir(exist_ok=True)

    cache_key = hashlib.sha256(f"{language}:{text}".encode()).hexdigest()
    cached_path = CACHE_DIR / f"{cache_key}.mp3"

    if cached_path.exists():
        return cached_path

    voice = VOICE_MAP.get(language, VOICE_MAP["es"])
    communicate = edge_tts.Communicate(text, voice)

    # Write to a temp file and atomically rename to avoid serving partial files.
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".mp3", dir=CACHE_DIR)
    tmp_path = Path(tmp_path_str)
    try:
        # Close the fd opened by mkstemp; edge_tts.save opens the file itself.
        import os
        os.close(tmp_fd)

        await communicate.save(str(tmp_path))
        tmp_path.rename(cached_path)
    except Exception:
        # Clean up the temp file on any failure to avoid corrupt cache entries.
        tmp_path.unlink(missing_ok=True)
        raise

    return cached_path
