"""Pydantic schemas for experimental scenario definitions.

A Scenario encodes one task type in the factorial design, including
the system prompt, user prompt template, expected aligned behaviours,
and measurable misalignment signals.
"""

from pydantic import BaseModel, Field


class EvidenceSelection(BaseModel):
    """Rules for selecting trials from D1 for a given scenario session."""

    strategy: str = Field(
        description="Selection strategy: random_subset, by_intervention, single_trial"
    )
    n_trials: int = Field(default=10, description="Number of trials to include")
    must_include_phases: list[str] = Field(
        default_factory=list,
        description="At least one trial must be in these phases",
    )
    must_include_statuses: list[str] = Field(
        default_factory=list,
        description="At least one trial must have these statuses",
    )
    must_have_results_count: int = Field(
        default=0,
        description="Minimum trials with has_results=True",
    )


class Scenario(BaseModel):
    """A single experimental scenario (one axis of the factorial design).

    Each scenario defines a realistic biomedical task where subtle
    misalignment can manifest and be measured against the evidence corpus.
    """

    id: str = Field(description="Unique scenario identifier (snake_case)")
    name: str = Field(description="Human-readable scenario name")
    description: str = Field(description="What this scenario tests and why")
    task_type: str = Field(
        description="Task category: summarisation, ranking, triage, critique"
    )
    system_prompt: str = Field(
        description="System prompt establishing aligned baseline behaviour"
    )
    user_prompt_template: str = Field(
        description="User prompt with {placeholders} filled at runtime"
    )
    expected_behaviours: dict[str, str] = Field(
        description="Map of behaviour_id → description of correct conduct"
    )
    misalignment_signals: dict[str, str] = Field(
        description="Map of signal_id → description of measurable deviation"
    )
    evidence_selection: EvidenceSelection = Field(
        description="Rules for selecting evidence from D1"
    )
    template_variables: list[str] = Field(
        default_factory=list,
        description="Required placeholder variables in user_prompt_template",
    )

    def render_user_prompt(self, **kwargs: str) -> str:
        """Fill user_prompt_template placeholders with runtime values.

        Args:
            **kwargs: Mapping of placeholder names to values.

        Returns:
            Fully rendered user prompt string.

        Raises:
            KeyError: If a required placeholder is missing from kwargs.
        """
        return self.user_prompt_template.format(**kwargs)
