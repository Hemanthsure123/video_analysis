"""openai_client.py — default report-LLM backend (OpenAI, gpt-4o). Phase 6."""

from __future__ import annotations

import config

from .base import LLMClient, LLMError

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        if not config.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY is not set (add it to .env).")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("openai SDK not installed (pip install openai).") from exc
        _CLIENT = OpenAI(api_key=config.OPENAI_API_KEY)
    return _CLIENT


class OpenAIClient(LLMClient):
    def complete(self, system: str, user: str, json_mode: bool = False,
                 temperature: float = 0.2) -> str:
        client = _client()
        kwargs = {
            "model": config.OPENAI_MODEL,
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
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return resp.choices[0].message.content or ""
