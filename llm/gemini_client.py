"""gemini_client.py — optional report backend (Google Gemini). Free tier.

Enable with LLM_BACKEND=gemini and GEMINI_API_KEY in .env.
"""

from __future__ import annotations

import config

from .base import LLMClient, LLMError

_MODEL = None


def _model(system: str):
    """Gemini takes the system prompt at model-construction time."""
    if not config.GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set (add it to .env).")
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise LLMError(
            "google-generativeai not installed (pip install google-generativeai)."
        ) from exc
    genai.configure(api_key=config.GEMINI_API_KEY)
    return genai.GenerativeModel(config.GEMINI_MODEL, system_instruction=system)


class GeminiClient(LLMClient):
    def complete(self, system: str, user: str, json_mode: bool = False,
                 temperature: float = 0.2) -> str:
        model = _model(system)
        gen_cfg = {
            "temperature": temperature,
            "response_mime_type": "application/json" if json_mode else "text/plain",
        }
        try:
            resp = model.generate_content(user, generation_config=gen_cfg)
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        return resp.text or ""
