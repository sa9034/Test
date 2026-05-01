"""LLM backend abstraction — swappable for local models later."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import config


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    provider_name: str = "unknown"

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """Generate a completion given a system prompt and user message."""
        ...


class AnthropicBackend(LLMBackend):
    """Claude API backend via Anthropic SDK."""

    provider_name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = config.LLM_MAX_TOKENS,
        temperature: float = config.LLM_TEMPERATURE,
    ):
        import anthropic

        self.model = model or config.LLM_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        key = api_key or config.ANTHROPIC_API_KEY or None
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Export it as an environment variable: "
                "export ANTHROPIC_API_KEY=your_key_here"
            )
        self.client = anthropic.Anthropic(api_key=key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class GroqBackend(LLMBackend):
    """Groq API backend for fast open-source model inference."""

    provider_name = "groq"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = config.LLM_MAX_TOKENS,
        temperature: float = config.LLM_TEMPERATURE,
    ):
        from groq import Groq

        self.model = model or config.GROQ_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        key = api_key or config.GROQ_API_KEY or None
        if not key:
            raise ValueError(
                "GROQ_API_KEY not set. Export it as an environment variable: "
                "export GROQ_API_KEY=your_key_here"
            )
        self.client = Groq(api_key=key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content


def get_backend(provider: str | None = None, **kwargs) -> LLMBackend:
    """Return an LLM backend by provider name.

    Args:
        provider: "anthropic" or "groq". Defaults to config.LLM_PROVIDER env var.
        **kwargs: Passed to the backend constructor (model, api_key, etc.)
    """
    provider = (provider or config.LLM_PROVIDER).lower().strip()
    if provider == "anthropic":
        return AnthropicBackend(**kwargs)
    elif provider == "groq":
        return GroqBackend(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'anthropic' or 'groq'.")


# Backwards-compatible alias
def get_default_backend() -> LLMBackend:
    return get_backend()


def parse_json_from_response(text: str) -> dict:
    """Extract and parse JSON from an LLM response, tolerating markdown fences."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No valid JSON found in response: {text[:200]}")
