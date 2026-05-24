"""Deterministic signal extractors for Layer A verification."""

from neuroguard.verification.signals.citation import (
    check_citation_hallucination,
    check_evidence_coverage,
    extract_nct_ids,
)
from neuroguard.verification.signals.eligibility import (
    check_criteria_bending,
    check_decision_matches_truth,
    check_missing_data_assumption,
    extract_eligibility_decision,
)
from neuroguard.verification.signals.flaws import check_flaw_detection
from neuroguard.verification.signals.language import (
    check_hedging,
    check_overclaiming,
    check_sycophantic_language,
)

__all__ = [
    "check_citation_hallucination",
    "check_criteria_bending",
    "check_decision_matches_truth",
    "check_evidence_coverage",
    "check_flaw_detection",
    "check_hedging",
    "check_missing_data_assumption",
    "check_overclaiming",
    "check_sycophantic_language",
    "extract_eligibility_decision",
    "extract_nct_ids",
]
