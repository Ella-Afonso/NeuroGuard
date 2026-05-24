"""Judge model adapters — abstraction over different judge providers.

The JudgeAdapter protocol decouples judge logic from any specific
provider. Tests use MockJudgeAdapter; production runs use OpenAIJudgeAdapter
or AnthropicJudgeAdapter (latter to be added when needed).

Real adapters lazy-import their SDKs so the judge module loads without
optional dependencies installed.
"""

from collections.abc import Callable
from typing import Protocol


class JudgeAdapter(Protocol):
    """Protocol for any object that can convert a judge prompt into a response.

    Implementations must be deterministic given the same arguments to be
    safe for use in tests. Real LLM-backed adapters are non-deterministic
    by nature; tests should always use MockJudgeAdapter.
    """

    model_name: str

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        """Send the prompt to the judge model and return its raw response."""
        ...


class MockJudgeAdapter:
    """In-memory judge adapter for tests.

    Returns either a fixed string response, or a response computed by a
    callable given the prompts. Never makes network calls.
    """

    def __init__(
        self,
        response: str | Callable[[str, str], str],
        model_name: str = "mock",
    ) -> None:
        """Initialise the mock adapter.

        Args:
            response: Either a fixed JSON string to return, or a callable
                ``(system_prompt, user_prompt) -> str`` for dynamic responses.
            model_name: Name to record in the rubric's ``judge_model`` field.
        """
        self._response = response
        self.model_name = model_name

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        if callable(self._response):
            return self._response(system_prompt, user_prompt)
        return self._response


class OpenAIJudgeAdapter:
    """Judge adapter backed by an OpenAI chat completion call.

    Uses ``response_format={"type": "json_object"}`` to guarantee JSON
    output. Lazy-imports the openai SDK so this module loads even when
    ``openai`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> None:
        """Initialise the OpenAI adapter.

        Args:
            model_name: OpenAI model identifier (e.g. ``gpt-4o-mini``).
            api_key: API key. If None, falls back to the OPENAI_API_KEY env var.
            max_tokens: Cap on judge response length.
            temperature: Sampling temperature; default 0 for reproducibility.
        """
        self.model_name = model_name
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature

    def judge(self, system_prompt: str, user_prompt: str) -> str:
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
            response_format={"type": "json_object"},
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        content = completion.choices[0].message.content
        return content or ""
