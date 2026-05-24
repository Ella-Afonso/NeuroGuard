"""Pydantic schemas for sessions — the unit of analysis."""

from typing import Any

from pydantic import BaseModel, Field

from neuroguard.judge.schema import JudgeRubric
from neuroguard.verification.schema import VerificationResult


class SessionCell(BaseModel):
    """Coordinates of one factorial cell.

    Identifies a unique combination of (scenario, pressure, monitoring,
    model, seed). Two sessions with the same cell coordinates should
    produce identical inputs but may produce different responses if the
    model under test is non-deterministic.
    """

    scenario_id: str = Field(description="Scenario identifier")
    pressure_id: str = Field(description="Pressure condition identifier")
    monitoring_id: str = Field(description="Monitoring condition identifier")
    model_name: str = Field(description="Model under test identifier")
    seed: int = Field(description="Seed used for context selection")


class Session(BaseModel):
    """One complete experimental session — the unit of downstream analysis.

    Bundles cell coordinates, the prompts as actually sent, the agent's
    response, both verification layers, and runtime metadata. Sessions
    are persisted as JSONL records and consumed by the analysis pipeline.
    """

    session_id: str = Field(description="Stable identifier for this session")
    cell: SessionCell = Field(description="Factorial cell coordinates")

    # Inputs
    system_prompt: str = Field(description="System prompt as sent to the model")
    user_prompt: str = Field(description="Rendered user prompt as sent to the model")

    # Output
    agent_response: str = Field(description="Raw response from the model under test")

    # Verification
    layer_a: VerificationResult = Field(description="Layer A programmatic verification")
    layer_b: JudgeRubric | None = Field(
        default=None,
        description="Layer B LLM-judge rubric (optional)",
    )

    # Metadata
    duration_seconds: float = Field(description="Wall-clock duration of the session")
    started_at: str = Field(description="ISO timestamp when the session started")
    completed_at: str = Field(description="ISO timestamp when the session completed")

    # Audit trail of which evidence/patient/proposal were used
    context_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Selected NCT IDs, patient_id, proposal_id, etc.",
    )

    @staticmethod
    def make_session_id(cell: SessionCell) -> str:
        """Compose a stable, human-readable session ID from cell coords."""
        return (
            f"{cell.scenario_id}__{cell.pressure_id}__{cell.monitoring_id}"
            f"__{cell.model_name}__seed{cell.seed}"
        )
