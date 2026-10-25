"""
base.py — the LLM backend interface.

Implementations:
  - openai_client.OpenAIClient  (default)
  - gemini_client.GeminiClient  (optional)
  - groq_client.GroqClient      (optional)

`get_client()` returns the backend selected by config.LLM_BACKEND so the rest of
the codebase never imports a concrete provider directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(Exception):
    """User-facing LLM failure (missing key, API error, etc.)."""


class LLMClient(ABC):
    """Minimal text-in / text-out interface used by report.py."""

    @abstractmethod
    def complete(self, system: str, user: str, json_mode: bool = False,
                 temperature: float = 0.2) -> str:
        """Return the model's text response (raw JSON string if json_mode)."""
        raise NotImplementedError


def get_client() -> LLMClient:
    """Return the LLMClient implementation chosen in config.LLM_BACKEND."""
    import config

    backend = (config.LLM_BACKEND or "openai").lower()
    if backend == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient()
    if backend == "gemini":
        from .gemini_client import GeminiClient  # type: ignore
        return GeminiClient()
    if backend == "groq":
        from .groq_client import GroqClient  # type: ignore
        return GroqClient()
    raise LLMError(f"Unknown LLM_BACKEND: {backend!r}")
