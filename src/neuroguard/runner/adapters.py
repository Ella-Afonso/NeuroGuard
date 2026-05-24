"""Model-under-test adapters.

Separates the model under test from the judge (Layer B). Provides a
mock adapter for tests and lazy-imported real adapters for OpenAI and
Anthropic. Adapters expose a uniform ``complete(system, user) -> str``
interface so the runner is provider-agnostic.
"""

from collections.abc import Callable
from typing import Protocol


class ModelAdapter(Protocol):
    """Uniform interface for any chat model used as the model under test."""

    model_name: str

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send the prompts to the model and return its response."""
        ...


class MockModelAdapter:
    """In-memory model adapter for tests and dry runs.

    Returns either a fixed string response, or a response computed by a
    callable given the prompts. Never makes network calls.
    """

    def __init__(
        self,
        response: str | Callable[[str, str], str],
        model_name: str = "mock-model",
    ) -> None:
        """Initialise the mock adapter.

        Args:
            response: Either a fixed string to return, or a callable
                ``(system_prompt, user_prompt) -> str`` for dynamic responses.
            model_name: Name to record in the session's ``model_name`` field.
        """
        self._response = response
        self.model_name = model_name

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if callable(self._response):
            return self._response(system_prompt, user_prompt)
        return self._response


class OpenAIModelAdapter:
    """Model adapter backed by an OpenAI chat completion call.

    Lazy-imports the openai SDK so this module loads even when
    ``openai`` is not installed. Default ``temperature=0.7`` for
    realistic deployment-like sampling (the JUDGE uses 0.0 for
    reproducibility, but the model under test is deliberately stochastic).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            msg = (
                "openai package not installed. Install with: "
                "pip install neuroguard[providers]"
            )
            raise ImportError(msg) from e

        client = OpenAI(api_key=self._api_key)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        content = completion.choices[0].message.content
        return content or ""


class AnthropicModelAdapter:
    """Model adapter backed by an Anthropic messages call.

    Lazy-imports the anthropic SDK. Default ``temperature=0.7`` for
    realistic deployment-like sampling.
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-latest",
        *,
        api_key: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            msg = (
                "anthropic package not installed. Install with: "
                "pip install neuroguard[providers]"
            )
            raise ImportError(msg) from e

        client = Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        # Anthropic returns a list of content blocks; concatenate text blocks
        parts = [
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        ]
        return "".join(parts)
