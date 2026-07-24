"""LLM service using claude -p subprocess for language tutoring."""

import asyncio
import json

from app.config import settings


async def chat(message: str, language: str, history: list[dict]) -> dict:
    """Send a message to claude -p for language tutoring.

    Args:
        message: The user's message.
        language: Target language code (e.g., "es").
        history: Previous conversation messages.

    Returns:
        Dict with "reply" and optional "suggestions".
    """
    # TODO: Implement claude -p subprocess call
    # Build prompt with conversation context
    # system_prompt = f"You are a {language} language tutor. ..."
    # prompt = build_prompt(system_prompt, history, message)
    # result = await asyncio.create_subprocess_exec(
    #     settings.claude_command, "-p", prompt,
    #     stdout=asyncio.subprocess.PIPE,
    #     stderr=asyncio.subprocess.PIPE,
    # )
    # stdout, _ = await result.communicate()
    # return {"reply": stdout.decode(), "suggestions": []}
    return {
        "reply": f"(stub) Response to: {message}",
        "suggestions": [],
    }
