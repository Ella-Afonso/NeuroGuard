"""Pydantic schema for pressure conditions."""

from typing import Literal

from pydantic import BaseModel, Field

PressureType = Literal[
    "none",
    "deadline",
    "authority",
    "peer_disagreement",
    "sunk_cost",
    "emotional_appeal",
    "hidden_instruction",
]

Severity = Literal["low", "medium", "high"]


class Pressure(BaseModel):
    """A single pressure condition applied to a scenario's prompts.

    Pressures layer on top of a scenario's baseline prompts to create
    realistic conditions where misalignment may emerge. The pressure
    metadata pre-registers which misalignment signals are expected to
    elevate, enabling hypothesis-driven analysis rather than post-hoc
    fishing.
    """

    id: str = Field(description="Unique pressure identifier (snake_case)")
    name: str = Field(description="Human-readable pressure name")
    description: str = Field(description="What this pressure tests and why")
    pressure_type: PressureType = Field(description="Pressure category")
    severity: Severity = Field(description="Estimated pressure intensity")

    # Application: how this pressure modifies a scenario's prompts.
    # Empty strings are no-ops.
    system_prompt_addition: str = Field(
        default="",
        description="Text appended to the scenario's system prompt",
    )
    user_prompt_prefix: str = Field(
        default="",
        description="Text prepended to the scenario's user prompt template",
    )
    user_prompt_suffix: str = Field(
        default="",
        description="Text appended to the scenario's user prompt template",
    )

    # Pre-registered hypotheses for hypothesis-driven analysis.
    expected_misalignment_signals: list[str] = Field(
        default_factory=list,
        description="Misalignment signal IDs expected to elevate under this pressure",
    )
    safety_review: str = Field(
        description="Why this pressure is ethically acceptable in this benchmark",
    )
