"""groq_client.py — optional report backend (Groq, Llama 3.3 70B). Free tier.

Enable with LLM_BACKEND=groq and GROQ_API_KEY in .env.
"""

from __future__ import annotations

import config

from .base import LLMClient, LLMError

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set (add it to .env).")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMError("groq SDK not installed (pip install groq).") from exc
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


class GroqClient(LLMClient):
    def complete(self, system: str, user: str, json_mode: bool = False,
                 temperature: float = 0.2) -> str:
        client = _client()
        kwargs = {
            "model": config.GROQ_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc
        return resp.choices[0].message.content or ""
