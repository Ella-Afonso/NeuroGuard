"""Configuration management for NeuroGuard.

Loads settings from environment variables (.env) and validates them.
Uses pydantic-settings for type coercion and validation.

Usage:
    from neuroguard.config import settings
    print(settings.log_level)
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings loaded from environment and .env file.

    Environment variables use the NEUROGUARD_ prefix for project-specific
    settings. Provider API keys use their standard names (no prefix).
    """

    # ---- Provider API keys (no prefix — standard env var names) ----
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
        description="Anthropic API key",
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key",
    )
    local_model_url: str | None = Field(
        default=None,
        validation_alias="LOCAL_MODEL_URL",
        description="Local model endpoint (e.g. Ollama, vLLM)",
    )

    # ---- Paths (NEUROGUARD_ prefix) ----
    data_dir: Path = Field(
        default=Path("data"),
        description="Root data directory",
    )
    results_dir: Path = Field(
        default=Path("results"),
        description="Root results directory",
    )

    # ---- Logging ----
    log_level: str = Field(default="INFO", description="Logging level")

    # ---- Experiment guardrails ----
    max_concurrent: int = Field(
        default=5,
        description="Max concurrent API requests",
    )
    cost_limit_usd: float = Field(
        default=10.0,
        description="Hard USD cost ceiling per experiment run",
    )

    model_config = {
        "env_prefix": "NEUROGUARD_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


# Module-level singleton.
# Import as: `from neuroguard.config import settings`
settings = Settings()
