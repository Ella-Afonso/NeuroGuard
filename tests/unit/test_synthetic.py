"""Tests for the synthetic content module — schemas, loaders, samplers."""

import pytest
from pydantic import ValidationError

from neuroguard.synthetic.loader import (
    DEFAULT_PATIENTS_PATH,
    DEFAULT_PROPOSALS_PATH,
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

# ---- Schema tests ----


class TestPatientProfileSchema:
    def test_minimal_valid_profile(self):
        """A profile with only required fields validates."""
        p = PatientProfile(
            patient_id="PT999",
            description="Test",
            age=70,
            sex="F",
            condition_status="Mild AD",
            eligibility_truth="ELIGIBLE",
            truth_rationale="Meets all criteria",
        )
        assert p.patient_id == "PT999"
        assert p.cdr_score is None
        assert p.comorbidities == []

    def test_age_out_of_range_rejected(self):
        """Age outside 0-120 raises ValidationError."""
        with pytest.raises(ValidationError):
            PatientProfile(
                patient_id="PT999",
                description="Test",
                age=200,
                sex="F",
                condition_status="x",
                eligibility_truth="ELIGIBLE",
                truth_rationale="x",
            )

    def test_invalid_eligibility_truth_rejected(self):
        """eligibility_truth must be one of the three literal values."""
        with pytest.raises(ValidationError):
            PatientProfile(
                patient_id="PT999",
                description="Test",
                age=70,
                sex="F",
                condition_status="x",
                eligibility_truth="MAYBE",  # invalid
                truth_rationale="x",
            )

    def test_to_prompt_text_includes_key_fields(self):
        """Rendered prompt text includes age, sex, diagnosis."""
        p = PatientProfile(
            patient_id="PT001",
            description="Test",
            age=72,
            sex="F",
            condition_status="Mild AD",
            cdr_score=1.0,
            mmse_score=22,
            comorbidities=["HTN"],
            medications=["Donepezil"],
            lab_values=LabValues(egfr=75.0),
            eligibility_truth="ELIGIBLE",
            truth_rationale="Meets all criteria",
        )
        text = p.to_prompt_text()
        assert "PT001" in text
        assert "72" in text
        assert "Mild AD" in text
        assert "CDR" in text
        assert "MMSE" in text
        assert "Donepezil" in text
        assert "75" in text


class TestProposalSchema:
    def test_proposal_with_flaws(self):
        """A proposal with planted flaws validates."""
        p = ResearchProposal(
            proposal_id="PR999",
            title="Test Proposal",
            text="Body text",
            planted_flaws=[
                Flaw(
                    flaw_id="PR999_F1",
                    flaw_type="underpowered",
                    severity="high",
                    description="n too small",
                    detectability="explicit",
                )
            ],
        )
        assert len(p.planted_flaws) == 1
        assert p.planted_flaws[0].flaw_type == "underpowered"

    def test_invalid_flaw_type_rejected(self):
        """flaw_type must be from the FlawType literal."""
        with pytest.raises(ValidationError):
            Flaw(
                flaw_id="X",
                flaw_type="not_a_real_flaw",
                severity="high",
                description="x",
                detectability="explicit",
            )


# ---- Loader tests ----


class TestLoaders:
    def test_load_patient_profiles(self):
        """Default patient pool loads and validates."""
        profiles = load_patient_profiles()
        assert len(profiles) >= 5
        # All have ground truth labels
        for p in profiles:
            assert p.eligibility_truth in {"ELIGIBLE", "NOT_ELIGIBLE", "INSUFFICIENT"}
            assert p.truth_rationale.strip()

    def test_load_research_proposals(self):
        """Default proposal pool loads and validates."""
        proposals = load_research_proposals()
        assert len(proposals) >= 3
        # Every proposal has at least one planted flaw
        for p in proposals:
            assert len(p.planted_flaws) >= 1
            for flaw in p.planted_flaws:
                assert flaw.detection_keywords, (
                    f"Flaw {flaw.flaw_id} missing detection keywords"
                )

    def test_default_paths_exist(self):
        """Documented default YAML paths exist."""
        assert DEFAULT_PATIENTS_PATH.exists()
        assert DEFAULT_PROPOSALS_PATH.exists()

    def test_unique_patient_ids(self):
        """Patient IDs are unique across the pool."""
        profiles = load_patient_profiles()
        ids = [p.patient_id for p in profiles]
        assert len(ids) == len(set(ids))

    def test_unique_proposal_ids(self):
        """Proposal IDs are unique across the pool."""
        proposals = load_research_proposals()
        ids = [p.proposal_id for p in proposals]
        assert len(ids) == len(set(ids))

    def test_truth_label_coverage(self):
        """Patient pool covers all three truth labels for design balance."""
        profiles = load_patient_profiles()
        labels = {p.eligibility_truth for p in profiles}
        assert "ELIGIBLE" in labels
        assert "NOT_ELIGIBLE" in labels
        assert "INSUFFICIENT" in labels


# ---- Sampler tests ----


class TestSamplers:
    def test_sample_patient_deterministic(self):
        """Same seed → same patient."""
        pool = load_patient_profiles()
        a = sample_patient(pool, seed=42)
        b = sample_patient(pool, seed=42)
        assert a.patient_id == b.patient_id

    def test_sample_patient_different_seeds(self):
        """Different seeds typically produce different patients."""
        pool = load_patient_profiles()
        # Try several seed pairs — at least one pair should differ
        results = {sample_patient(pool, seed=s).patient_id for s in range(20)}
        assert len(results) >= 2

    def test_sample_proposal_deterministic(self):
        """Same seed → same proposal."""
        pool = load_research_proposals()
        a = sample_proposal(pool, seed=7)
        b = sample_proposal(pool, seed=7)
        assert a.proposal_id == b.proposal_id

    def test_sample_empty_pool_raises(self):
        """Sampling from empty pool raises ValueError."""
        with pytest.raises(ValueError):
            sample_patient([], seed=0)
        with pytest.raises(ValueError):
            sample_proposal([], seed=0)
