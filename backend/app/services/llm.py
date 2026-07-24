"""LLM service using claude -p subprocess for language tutoring."""

import asyncio
import json
import logging
import os
import shutil

from app.config import settings

log = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "es": "Spanish",
    "hr": "Croatian",
    "de": "German",
    "zh": "Mandarin Chinese",
}

SYSTEM_PROMPT_TEMPLATE = """\
You are a friendly, patient {language_name} language tutor having a conversation \
with a learner. Your goals:

1. Conduct the conversation primarily in {language_name}, adjusting complexity \
to the learner's level.
2. When the learner makes a grammar or vocabulary mistake, gently correct it \
and explain briefly in English.
3. Introduce new vocabulary naturally and provide translations in parentheses \
when using words the learner might not know.
4. Keep responses concise — 1-3 sentences in {language_name}, plus a brief \
English note if needed.
5. At the end of each reply, suggest 2-3 short phrases the learner could say \
next, formatted as a JSON array on the last line, like:
   SUGGESTIONS: ["¿Cómo estás?", "Me llamo...", "¿Qué hora es?"]

If the learner writes in English, respond in {language_name} with an English \
translation. If they write in {language_name}, respond naturally in {language_name} \
and correct any errors.\
"""


def _find_claude() -> str:
    """Find the claude CLI binary."""
    # Check configured command first
    cmd = settings.claude_command
    if shutil.which(cmd):
        return cmd
    # Common install locations
    for candidate in ("claude", "/usr/local/bin/claude", os.path.expanduser("~/.claude/local/claude")):
        if shutil.which(candidate):
            return candidate
    return cmd  # Fall back to configured, let subprocess raise if missing


def _build_prompt(message: str, language: str, history: list[dict]) -> str:
    """Build the conversation prompt including history."""
    parts: list[str] = []

    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"Learner: {content}")
        else:
            parts.append(f"Tutor: {content}")

    parts.append(f"Learner: {message}")
    parts.append("Tutor:")

    return "\n".join(parts)


def _parse_suggestions(text: str) -> tuple[str, list[str]]:
    """Extract SUGGESTIONS JSON from the response, returning (reply, suggestions)."""
    lines = text.strip().split("\n")
    suggestions: list[str] = []
    reply_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SUGGESTIONS:"):
            try:
                json_part = stripped[len("SUGGESTIONS:"):].strip()
                parsed = json.loads(json_part)
                if isinstance(parsed, list):
                    suggestions = [str(s) for s in parsed]
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, treat as normal text
                reply_lines.append(line)
        else:
            reply_lines.append(line)

    reply = "\n".join(reply_lines).strip()
    return reply, suggestions


async def chat(message: str, language: str, history: list[dict]) -> dict:
    """Send a message to claude -p for language tutoring.

    Args:
        message: The user's message.
        language: Target language code (e.g., "es").
        history: Previous conversation messages.

    Returns:
        Dict with "reply" and "suggestions".
    """
    language_name = LANGUAGE_NAMES.get(language, language)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language_name=language_name)
    prompt = _build_prompt(message, language, history)

    claude_cmd = _find_claude()

    # Strip CLAUDECODE env var to allow nested invocation
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    try:
        proc = await asyncio.create_subprocess_exec(
            claude_cmd, "-p", prompt,
            "--system-prompt", system_prompt,
            "--output-format", "json",
            "--no-session-persistence",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip()
            log.error("claude -p failed (exit %d): %s", proc.returncode, err)
            return {
                "reply": f"(Tutor is unavailable: {err[:200]})",
                "suggestions": [],
            }

        raw = stdout.decode().strip()

        # Parse JSON envelope from --output-format json
        try:
            envelope = json.loads(raw)
            text = envelope.get("result", raw)
        except json.JSONDecodeError:
            text = raw

        reply, suggestions = _parse_suggestions(text)

        # Provide default suggestions if none were generated
        if not suggestions:
            suggestions = _default_suggestions(language)

        return {"reply": reply, "suggestions": suggestions}

    except FileNotFoundError:
        log.error("claude command not found: %s", claude_cmd)
        return {
            "reply": "(Tutor unavailable — claude CLI not found. Install from https://docs.anthropic.com/en/docs/claude-code)",
            "suggestions": [],
        }
    except Exception as e:
        log.error("claude -p error: %s", e)
        return {
            "reply": f"(Tutor error: {e})",
            "suggestions": [],
        }


def _default_suggestions(language: str) -> list[str]:
    """Fallback suggestions by language."""
    defaults = {
        "es": ["¿Cómo estás?", "¿Puedes repetir?", "No entiendo"],
        "hr": ["Kako si?", "Možeš ponoviti?", "Ne razumijem"],
        "de": ["Wie geht es dir?", "Kannst du wiederholen?", "Ich verstehe nicht"],
        "zh": ["你好吗？", "请再说一遍", "我不明白"],
    }
    return defaults.get(language, defaults["es"])
