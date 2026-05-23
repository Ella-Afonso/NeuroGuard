"""Synthetic content for triage and critique scenarios.

Provides deterministic, seeded samplers over hand-curated pools of:
- Patient profiles (for eligibility_triage)
- Research proposals (for proposal_critique)

All content is synthetic. No real patient data or real research proposals.
"""

from neuroguard.synthetic.loader import (
    load_patient_profiles,
    load_research_proposals,
    sample_patient,
    sample_proposal,
)
from neuroguard.synthetic.schema import (
    Flaw,
    LabValues,
    PatientProfile,
    ResearchProposal,
)

__all__ = [
    "Flaw",
    "LabValues",
    "PatientProfile",
    "ResearchProposal",
    "load_patient_profiles",
    "load_research_proposals",
    "sample_patient",
    "sample_proposal",
]
