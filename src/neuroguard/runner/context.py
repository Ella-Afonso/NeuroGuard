"""Build per-scenario task context from seeded selections of evidence/patient/proposal."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from neuroguard.evidence.fetch import load_cached_trials
from neuroguard.evidence.schema import TrialRecord
from neuroguard.runner.evidence_select import select_evidence
from neuroguard.scenarios.schema import Scenario
from neuroguard.synthetic.loader import sample_patient, sample_proposal
from neuroguard.synthetic.schema import PatientProfile, ResearchProposal

DEFAULT_TRIAL_CACHE = Path("data/raw/trials.jsonl")


class TaskContext(BaseModel):
    """Bundle of runtime context for one session.

    Carries the rendered prompt variables AND the structured objects
    Layer A needs for verification (NCT IDs, patient, proposal). Also
    records selection metadata for audit.
    """

    variables: dict[str, str] = Field(
        description="Variables to fill the scenario's user_prompt_template"
    )
    evidence_nct_ids: set[str] | None = Field(
        default=None,
        description="NCT IDs in the evidence shown to the agent (for Layer A)",
    )
    patient: PatientProfile | None = Field(
        default=None, description="Selected patient profile (triage scenarios)"
    )
    proposal: ResearchProposal | None = Field(
        default=None, description="Selected research proposal (critique scenarios)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Selection audit (which IDs were chosen)",
    )


def render_trial_block(trial: TrialRecord) -> str:
    """Render a single trial as a readable text block for the agent prompt."""
    header_bits = [
        trial.phase or "unknown phase",
        trial.overall_status or "unknown status",
    ]
    if trial.enrollment is not None:
        header_bits.append(f"n={trial.enrollment}")
    parts = [f"Trial {trial.nct_id} ({', '.join(header_bits)}):"]
    parts.append(f"  Title: {trial.brief_title}")
    if trial.conditions:
        parts.append(f"  Conditions: {', '.join(trial.conditions)}")
    if trial.interventions:
        parts.append(f"  Interventions: {', '.join(trial.interventions)}")
    if trial.primary_outcomes:
        parts.append(f"  Primary outcomes: {'; '.join(trial.primary_outcomes)}")
    if trial.lead_sponsor:
        parts.append(f"  Sponsor: {trial.lead_sponsor}")
    parts.append(f"  Has posted results: {'yes' if trial.has_results else 'no'}")
    return "\n".join(parts)


def render_evidence_block(trials: list[TrialRecord]) -> str:
    """Render a list of trials as a single text block."""
    return "\n\n".join(render_trial_block(t) for t in trials)


def load_trial_pool(cache_path: Path | None = None) -> list[TrialRecord]:
    """Load the cached D1 trial pool, raising a clear error if missing."""
    if cache_path is None:
        cache_path = DEFAULT_TRIAL_CACHE
    pool = load_cached_trials(cache_path)
    if pool is None:
        msg = (
            f"No cached trials found at {cache_path}. "
            "Run `neuroguard fetch-evidence` first to populate the cache."
        )
        raise FileNotFoundError(msg)
    return pool


def _first_or(items: list[str], default: str) -> str:
    return items[0] if items else default


def build_task_context(
    scenario: Scenario,
    seed: int,
    *,
    trial_pool: list[TrialRecord] | None = None,
    patient_pool: list[PatientProfile] | None = None,
    proposal_pool: list[ResearchProposal] | None = None,
) -> TaskContext:
    """Build runtime context for one session of the given scenario.

    Args:
        scenario: The scenario being run.
        seed: Seed for deterministic selection.
        trial_pool: D1 trial pool (required for summarisation, ranking, triage, critique).
        patient_pool: Patient profiles (required for triage).
        proposal_pool: Research proposals (required for critique).

    Returns:
        A TaskContext with rendered variables and Layer A inputs.

    Raises:
        ValueError: If a required pool is missing for the scenario type.
    """
    task = scenario.task_type

    if task == "summarisation":
        if trial_pool is None:
            msg = "trial_pool required for summarisation scenarios"
            raise ValueError(msg)
        trials = select_evidence(scenario.evidence_selection, trial_pool, seed)
        evidence_block = render_evidence_block(trials)
        variables = {
            "intervention": _first_or(
                trials[0].interventions if trials else [], "the intervention"
            ),
            "condition": _first_or(
                trials[0].conditions if trials else [], "the condition"
            ),
            "evidence_block": evidence_block,
        }
        return TaskContext(
            variables=variables,
            evidence_nct_ids={t.nct_id for t in trials},
            metadata={"selected_nct_ids": [t.nct_id for t in trials]},
        )

    if task == "ranking":
        if trial_pool is None:
            msg = "trial_pool required for ranking scenarios"
            raise ValueError(msg)
        trials = select_evidence(scenario.evidence_selection, trial_pool, seed)
        evidence_block = render_evidence_block(trials)
        intervention_hint = _first_or(
            trials[0].interventions if trials else [], "the intervention"
        )
        condition_hint = _first_or(
            trials[0].conditions if trials else [], "the condition"
        )
        clinical_question = (
            f"Which trial(s) of {intervention_hint} for {condition_hint} "
            "would you prioritise for a systematic review, and why?"
        )
        variables = {
            "clinical_question": clinical_question,
            "evidence_block": evidence_block,
        }
        return TaskContext(
            variables=variables,
            evidence_nct_ids={t.nct_id for t in trials},
            metadata={"selected_nct_ids": [t.nct_id for t in trials]},
        )

    if task == "triage":
        if trial_pool is None or patient_pool is None:
            msg = "trial_pool and patient_pool required for triage scenarios"
            raise ValueError(msg)
        trials = select_evidence(scenario.evidence_selection, trial_pool, seed)
        if not trials:
            msg = "No trial selected for triage scenario"
            raise ValueError(msg)
        trial = trials[0]
        patient = sample_patient(patient_pool, seed)
        variables = {
            "trial_title": trial.brief_title,
            "nct_id": trial.nct_id,
            "trial_phase": trial.phase or "unknown",
            "trial_status": trial.overall_status or "unknown",
            "eligibility_criteria": (
                trial.eligibility_criteria
                or "(no specific criteria provided in the source record)"
            ),
            "patient_profile": patient.to_prompt_text(),
        }
        return TaskContext(
            variables=variables,
            patient=patient,
            metadata={
                "selected_nct_id": trial.nct_id,
                "patient_id": patient.patient_id,
                "patient_truth": patient.eligibility_truth,
            },
        )

    if task == "critique":
        if trial_pool is None or proposal_pool is None:
            msg = "trial_pool and proposal_pool required for critique scenarios"
            raise ValueError(msg)
        proposal = sample_proposal(proposal_pool, seed)
        trials = select_evidence(scenario.evidence_selection, trial_pool, seed)
        evidence_block = render_evidence_block(trials)
        variables = {
            "proposal_title": proposal.title,
            "proposal_text": proposal.text,
            "evidence_block": evidence_block,
        }
        return TaskContext(
            variables=variables,
            proposal=proposal,
            evidence_nct_ids={t.nct_id for t in trials},
            metadata={
                "proposal_id": proposal.proposal_id,
                "selected_nct_ids": [t.nct_id for t in trials],
            },
        )

    msg = f"Unknown scenario task_type: {task}"
    raise ValueError(msg)
