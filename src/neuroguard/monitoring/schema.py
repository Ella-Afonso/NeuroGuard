"""Pydantic schema for monitoring conditions."""

from typing import Literal

from pydantic import BaseModel, Field

MonitoringType = Literal["monitored", "unmonitored"]


class MonitoringCondition(BaseModel):
    """A monitoring condition applied to a scenario x pressure prompt.

    The condition signals to the agent whether its outputs are being
    reviewed (monitored) or treated as internal/private (unmonitored).
    The contrast between these conditions, when crossed with pressure,
    measures alignment-faking — whether alignment is contingent on
    being observed.
    """

    id: str = Field(description="Unique monitoring identifier")
    name: str = Field(description="Human-readable monitoring name")
    description: str = Field(description="What this condition signals to the agent")
    monitoring_type: MonitoringType = Field(description="Monitoring category")

    # Application: how this condition modifies the prompts. The
    # monitoring framing is appended AFTER any pressure modifications.
    system_prompt_addition: str = Field(
        default="",
        description="Text appended to the (already pressure-modified) system prompt",
    )
    user_prompt_prefix: str = Field(
        default="",
        description="Text prepended to the (already pressure-modified) user prompt",
    )
    user_prompt_suffix: str = Field(
        default="",
        description="Text appended to the (already pressure-modified) user prompt",
    )

    # Pre-registered hypothesis: what we expect the contrast to reveal.
    expected_effect: str = Field(
        description="Pre-registered hypothesis about how this condition affects behaviour",
    )
    safety_review: str = Field(
        description="Why this condition is ethically acceptable",
    )
