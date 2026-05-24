"""Experiment configuration — YAML-driven factorial sweep definitions.

An experiment config specifies which scenarios, pressures, monitoring
conditions, models, and seeds to sweep. The config loader validates
the YAML, resolves all references to loaded objects, and builds the
adapters needed for `run_batch`.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from neuroguard.logging_setup import get_logger

logger = get_logger(__name__)


class ModelSpec(BaseModel):
    """Specification for one model to test or use as judge."""

    provider: str = Field(description="Provider: openai | anthropic | mock")
    name: str = Field(description="Model name (e.g. gpt-4o-mini)")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=1500, description="Max tokens for response")


class JudgeSpec(BaseModel):
    """Specification for the Layer B judge model (optional)."""

    provider: str = Field(default="none", description="Provider: openai | mock | none")
    name: str = Field(default="", description="Model name")
    temperature: float = Field(
        default=0.0, description="Sampling temperature (0 for repro)"
    )
    max_tokens: int = Field(default=800, description="Max tokens for judge response")


class ExperimentConfig(BaseModel):
    """Full experiment configuration loaded from YAML.

    Defines the complete factorial design for one experiment run.
    The total number of cells = len(scenarios) * len(pressures) *
    len(monitoring) * len(models) * len(seeds).
    """

    name: str = Field(description="Human-readable experiment name")
    description: str = Field(
        default="", description="Experiment description/hypothesis"
    )

    scenarios: list[str] = Field(description="Scenario IDs to include")
    pressures: list[str] = Field(description="Pressure IDs to include")
    monitoring: list[str] = Field(description="Monitoring condition IDs to include")
    models: list[ModelSpec] = Field(description="Models under test")
    judge: JudgeSpec = Field(default_factory=JudgeSpec, description="Judge config")
    seeds: list[int] = Field(description="Seeds for deterministic context selection")

    output_path: str = Field(
        default="data/sessions/experiment.jsonl",
        description="Path to write session JSONL output",
    )
    trials_path: str = Field(
        default="data/raw/trials.jsonl",
        description="Path to cached D1 trial pool",
    )

    def total_cells(self) -> int:
        """Total number of sessions this experiment will produce."""
        return (
            len(self.scenarios)
            * len(self.pressures)
            * len(self.monitoring)
            * len(self.models)
            * len(self.seeds)
        )

    def cost_estimate_usd(self, cost_per_session: float = 0.005) -> float:
        """Rough cost estimate at given rate per session (default for gpt-4o-mini)."""
        return self.total_cells() * cost_per_session


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment config from YAML.

    Args:
        path: Path to the experiment YAML file.

    Returns:
        A validated ExperimentConfig.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is invalid or fails validation.
    """
    if not path.exists():
        msg = f"Experiment config not found: {path}"
        raise FileNotFoundError(msg)

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if raw is None:
        msg = f"Experiment config is empty: {path}"
        raise ValueError(msg)

    config = ExperimentConfig(**raw)
    logger.info(
        "experiment_config_loaded",
        name=config.name,
        total_cells=config.total_cells(),
        cost_estimate=f"${config.cost_estimate_usd():.2f}",
    )
    return config
