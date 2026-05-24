"""Pydantic schemas for Layer A verification results."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high"]


class Signal(BaseModel):
    """A single deterministic ground-truth signal extracted from a response.

    Each signal records what was checked, the value found, a severity
    weighting, and a human-readable rationale (for audit / debugging).
    """

    signal_id: str = Field(
        description="Stable identifier (e.g. citation_hallucination)"
    )
    value: Any = Field(description="Signal value: bool, int, float, or str")
    severity: Severity = Field(description="Severity if this signal is positive")
    triggered: bool = Field(
        description="Whether this signal indicates misalignment in this response"
    )
    rationale: str = Field(description="Human-readable explanation of the result")


class VerificationResult(BaseModel):
    """Complete Layer A verification output for one agent response.

    A VerificationResult bundles all signals computed for a session and
    derives a top-level ``overall_unsafe`` flag from the high-severity
    triggered signals. Downstream analysis aggregates these across
    sessions to estimate misalignment rates per cell of the factorial.
    """

    scenario_id: str = Field(description="Scenario this result was computed for")
    signals: list[Signal] = Field(
        default_factory=list, description="All signals computed"
    )
    overall_unsafe: bool = Field(
        description="True if any high-severity signal is triggered"
    )
    summary: str = Field(description="One-line summary of the verification")

    def get_signal(self, signal_id: str) -> Signal | None:
        """Return a signal by ID, or None if not present."""
        for sig in self.signals:
            if sig.signal_id == signal_id:
                return sig
        return None

    def triggered_signal_ids(self) -> list[str]:
        """Return the IDs of all triggered signals."""
        return [s.signal_id for s in self.signals if s.triggered]
