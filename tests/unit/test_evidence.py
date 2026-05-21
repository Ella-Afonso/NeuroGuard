"""Tests for the evidence corpus module — schema, parsing, and caching."""

import pytest
from pydantic import ValidationError

from neuroguard.evidence.fetch import load_cached_trials
from neuroguard.evidence.schema import TrialRecord, parse_study

# ---- Fixtures ----


@pytest.fixture
def raw_ctgov_study() -> dict:
    """A realistic CT.gov API v2 study response object."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "Test Study on Alzheimer's",
                "officialTitle": "A Randomized Test Study",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2020-01-15", "type": "ACTUAL"},
                "completionDateStruct": {"date": "2023-06-30", "type": "ACTUAL"},
            },
            "conditionsModule": {
                "conditions": ["Alzheimer Disease", "Dementia"],
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE3"],
                "enrollmentInfo": {"count": 200, "type": "ACTUAL"},
            },
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": "Aducanumab", "description": "..."},
                    {"type": "DRUG", "name": "Placebo", "description": "..."},
                ],
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {"measure": "CDR-SB Change", "timeFrame": "78 weeks"},
                ],
                "secondaryOutcomes": [
                    {"measure": "ADAS-Cog", "timeFrame": "78 weeks"},
                ],
            },
            "eligibilityModule": {
                "eligibilityCriteria": "Inclusion: Age 50-85...",
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Biogen", "class": "INDUSTRY"},
            },
        },
        "hasResults": True,
    }


# ---- Schema & parsing tests ----


class TestTrialRecord:
    def test_parse_valid_study(self, raw_ctgov_study):
        """Full CT.gov study parses into correct TrialRecord fields."""
        record = parse_study(raw_ctgov_study)
        assert record.nct_id == "NCT12345678"
        assert record.brief_title == "Test Study on Alzheimer's"
        assert record.official_title == "A Randomized Test Study"
        assert "Alzheimer Disease" in record.conditions
        assert "Aducanumab" in record.interventions
        assert "Placebo" in record.interventions
        assert record.phase == "PHASE3"
        assert record.overall_status == "COMPLETED"
        assert record.study_type == "INTERVENTIONAL"
        assert record.enrollment == 200
        assert "CDR-SB Change" in record.primary_outcomes
        assert "ADAS-Cog" in record.secondary_outcomes
        assert record.lead_sponsor == "Biogen"
        assert record.has_results is True
        assert record.start_date == "2020-01-15"
        assert record.completion_date == "2023-06-30"
        assert "NCT12345678" in record.source_url
        assert record.api_version == "v2"
        assert record.fetched_at  # non-empty ISO string

    def test_parse_minimal_study(self):
        """Parser handles a study with only required fields."""
        minimal = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000001",
                    "briefTitle": "Minimal Study",
                },
            },
        }
        record = parse_study(minimal)
        assert record.nct_id == "NCT00000001"
        assert record.brief_title == "Minimal Study"
        assert record.conditions == []
        assert record.interventions == []
        assert record.phase is None
        assert record.enrollment is None
        assert record.has_results is False

    def test_parse_empty_study(self):
        """Parser survives a completely empty study dict."""
        record = parse_study({})
        assert record.nct_id == "UNKNOWN"
        assert record.brief_title == ""

    def test_multi_phase(self):
        """Multiple phases are joined with comma separator."""
        study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT99999999",
                    "briefTitle": "Multi-phase",
                },
                "designModule": {
                    "phases": ["PHASE2", "PHASE3"],
                },
            },
        }
        record = parse_study(study)
        assert record.phase == "PHASE2, PHASE3"

    def test_record_json_round_trip(self, raw_ctgov_study):
        """TrialRecord serializes to JSON and deserializes back identically."""
        record = parse_study(raw_ctgov_study)
        json_str = record.model_dump_json()
        restored = TrialRecord.model_validate_json(json_str)
        assert restored.nct_id == record.nct_id
        assert restored.conditions == record.conditions
        assert restored.interventions == record.interventions
        assert restored.enrollment == record.enrollment

    def test_record_rejects_missing_required(self):
        """TrialRecord raises ValidationError without required fields."""
        with pytest.raises(ValidationError):
            TrialRecord()


# ---- Cache tests ----


class TestCache:
    def test_round_trip(self, raw_ctgov_study, tmp_path):
        """Write JSONL and read it back — records match."""
        record = parse_study(raw_ctgov_study)
        cache_path = tmp_path / "test_cache.jsonl"

        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

        loaded = load_cached_trials(cache_path)
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].nct_id == "NCT12345678"

    def test_missing_file_returns_none(self, tmp_path):
        """load_cached_trials returns None for a nonexistent file."""
        result = load_cached_trials(tmp_path / "nonexistent.jsonl")
        assert result is None

    def test_multiple_records(self, raw_ctgov_study, tmp_path):
        """Multiple records are written and read back correctly."""
        records = [parse_study(raw_ctgov_study) for _ in range(5)]
        cache_path = tmp_path / "multi.jsonl"

        with open(cache_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")

        loaded = load_cached_trials(cache_path)
        assert len(loaded) == 5
        assert all(r.nct_id == "NCT12345678" for r in loaded)

    def test_empty_lines_skipped(self, raw_ctgov_study, tmp_path):
        """Blank lines in JSONL are silently skipped."""
        record = parse_study(raw_ctgov_study)
        cache_path = tmp_path / "with_blanks.jsonl"

        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
            f.write("\n")
            f.write("  \n")
            f.write(record.model_dump_json() + "\n")

        loaded = load_cached_trials(cache_path)
        assert len(loaded) == 2
