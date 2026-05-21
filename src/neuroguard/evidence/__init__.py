"""Evidence corpus — ClinicalTrials.gov data ingestion and validation."""

from neuroguard.evidence.fetch import fetch_trials, load_cached_trials
from neuroguard.evidence.schema import TrialRecord, parse_study

__all__ = ["TrialRecord", "fetch_trials", "load_cached_trials", "parse_study"]
