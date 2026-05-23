"""Loaders and seeded samplers for synthetic patient profiles and proposals."""

import random
from pathlib import Path

import yaml

from neuroguard.logging_setup import get_logger
from neuroguard.synthetic.schema import PatientProfile, ResearchProposal

logger = get_logger(__name__)

DEFAULT_PATIENTS_PATH = Path("configs/synthetic/patient_profiles.yaml")
DEFAULT_PROPOSALS_PATH = Path("configs/synthetic/research_proposals.yaml")


def load_patient_profiles(path: Path | None = None) -> list[PatientProfile]:
    """Load and validate the patient profile pool from YAML.

    Args:
        path: Path to the YAML file. Defaults to
            configs/synthetic/patient_profiles.yaml.

    Returns:
        List of validated PatientProfile objects.
    """
    if path is None:
        path = DEFAULT_PATIENTS_PATH

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    profiles = [PatientProfile.model_validate(p) for p in raw["patients"]]
    logger.info("patient_profiles_loaded", count=len(profiles), path=str(path))
    return profiles


def load_research_proposals(path: Path | None = None) -> list[ResearchProposal]:
    """Load and validate the research proposal pool from YAML.

    Args:
        path: Path to the YAML file. Defaults to
            configs/synthetic/research_proposals.yaml.

    Returns:
        List of validated ResearchProposal objects.
    """
    if path is None:
        path = DEFAULT_PROPOSALS_PATH

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    proposals = [ResearchProposal.model_validate(p) for p in raw["proposals"]]
    logger.info("proposals_loaded", count=len(proposals), path=str(path))
    return proposals


def sample_patient(pool: list[PatientProfile], seed: int) -> PatientProfile:
    """Deterministically sample one patient profile from the pool.

    Args:
        pool: List of patient profiles to sample from.
        seed: Random seed for reproducibility.

    Returns:
        A single PatientProfile.

    Raises:
        ValueError: If the pool is empty.
    """
    if not pool:
        msg = "Cannot sample from empty patient profile pool"
        raise ValueError(msg)
    rng = random.Random(seed)
    return rng.choice(pool)


def sample_proposal(pool: list[ResearchProposal], seed: int) -> ResearchProposal:
    """Deterministically sample one research proposal from the pool.

    Args:
        pool: List of research proposals to sample from.
        seed: Random seed for reproducibility.

    Returns:
        A single ResearchProposal.

    Raises:
        ValueError: If the pool is empty.
    """
    if not pool:
        msg = "Cannot sample from empty research proposal pool"
        raise ValueError(msg)
    rng = random.Random(seed)
    return rng.choice(pool)
