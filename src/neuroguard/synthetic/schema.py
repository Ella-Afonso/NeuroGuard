"""Pydantic schemas for synthetic patient profiles and research proposals.

All content is synthetic; no real patient data. Each item carries
explicit ground-truth metadata used by Layer A programmatic verification.
"""

from typing import Literal

from pydantic import BaseModel, Field

EligibilityTruth = Literal["ELIGIBLE", "NOT_ELIGIBLE", "INSUFFICIENT"]

FlawType = Literal[
    "underpowered",
    "missing_control",
    "evidence_misrepresentation",
    "endpoint_mismatch",
    "conflict_of_interest",
    "inadequate_consent",
    "population_mismatch",
    "statistical_error",
    "ethical_concern",
]

Severity = Literal["low", "medium", "high"]
Detectability = Literal["explicit", "subtle"]


class LabValues(BaseModel):
    """Lab values relevant to AD trial eligibility decisions."""

    egfr: float | None = Field(default=None, description="eGFR (mL/min/1.73m²)")
    hemoglobin: float | None = Field(default=None, description="Hemoglobin (g/dL)")
    creatinine: float | None = Field(default=None, description="Creatinine (mg/dL)")
    alt: float | None = Field(default=None, description="ALT (U/L)")


class PatientProfile(BaseModel):
    """A synthetic patient profile for eligibility_triage scenarios.

    Designed to test whether the agent correctly applies stated trial
    eligibility criteria. Each profile has an explicit ground truth
    label and rationale.
    """

    patient_id: str = Field(description="Synthetic patient identifier (e.g. PT001)")
    description: str = Field(description="Short label for the test case")
    age: int = Field(ge=0, le=120, description="Age in years")
    sex: Literal["M", "F", "Other"] = Field(description="Sex")
    condition_status: str = Field(description="Condition diagnosis and severity")
    cdr_score: float | None = Field(
        default=None, ge=0, le=3, description="Clinical Dementia Rating global score"
    )
    mmse_score: int | None = Field(
        default=None, ge=0, le=30, description="Mini-Mental State Examination score"
    )
    comorbidities: list[str] = Field(
        default_factory=list, description="Active comorbid conditions"
    )
    medications: list[str] = Field(
        default_factory=list, description="Current medications"
    )
    lab_values: LabValues = Field(
        default_factory=LabValues, description="Relevant lab values"
    )
    notes: str = Field(default="", description="Additional relevant history")
    eligibility_truth: EligibilityTruth = Field(
        description="Ground truth eligibility for a typical Phase 3 AD trial"
    )
    truth_rationale: str = Field(
        description="Why this label — which specific criterion drives it"
    )

    def to_prompt_text(self) -> str:
        """Render the patient profile as readable text for the prompt."""
        lines = [
            f"Patient ID: {self.patient_id}",
            f"Age: {self.age} | Sex: {self.sex}",
            f"Diagnosis: {self.condition_status}",
        ]
        if self.cdr_score is not None:
            lines.append(f"CDR Global Score: {self.cdr_score}")
        if self.mmse_score is not None:
            lines.append(f"MMSE: {self.mmse_score}/30")
        if self.comorbidities:
            lines.append(f"Comorbidities: {', '.join(self.comorbidities)}")
        else:
            lines.append("Comorbidities: None reported")
        if self.medications:
            lines.append(f"Current medications: {', '.join(self.medications)}")
        else:
            lines.append("Current medications: None")

        labs = self.lab_values
        lab_parts = []
        if labs.egfr is not None:
            lab_parts.append(f"eGFR {labs.egfr}")
        if labs.hemoglobin is not None:
            lab_parts.append(f"Hb {labs.hemoglobin}")
        if labs.creatinine is not None:
            lab_parts.append(f"Cr {labs.creatinine}")
        if labs.alt is not None:
            lab_parts.append(f"ALT {labs.alt}")
        if lab_parts:
            lines.append(f"Lab values: {', '.join(lab_parts)}")
        else:
            lines.append("Lab values: Not reported")

        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


class Flaw(BaseModel):
    """A deliberately planted flaw in a research proposal.

    Used as ground truth for Layer A verification of proposal_critique
    scenarios — did the agent catch this specific issue?
    """

    flaw_id: str = Field(description="Unique flaw identifier (e.g. PR001_F1)")
    flaw_type: FlawType = Field(description="Category of flaw")
    severity: Severity = Field(description="How serious this flaw is")
    description: str = Field(description="What the flaw is and where in the proposal")
    detectability: Detectability = Field(
        description="explicit = surface-level; subtle = requires careful reading"
    )
    detection_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords/phrases that, if present in agent response, count as a catch",
    )


class ResearchProposal(BaseModel):
    """A synthetic research proposal for proposal_critique scenarios.

    Each proposal contains 1-3 deliberately planted flaws documented in
    `planted_flaws`. Used to measure whether the agent identifies them.
    """

    proposal_id: str = Field(description="Synthetic proposal identifier (e.g. PR001)")
    title: str = Field(description="Proposal title")
    text: str = Field(description="Full proposal body text")
    planted_flaws: list[Flaw] = Field(description="Documented flaws in this proposal")
    target_condition: str = Field(
        default="Alzheimer Disease",
        description="Condition this proposal targets",
    )
