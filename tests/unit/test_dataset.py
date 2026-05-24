"""Tests for dataset assembly and feature engineering."""

import json

import pandas as pd
import pytest

from neuroguard.dataset.assemble import (
    assemble_dataset,
    build_d2a,
    build_d2b,
    build_d2c,
    build_d2d,
    load_sessions_raw,
)
from neuroguard.dataset.features import engineer_features


def _make_session(
    session_id: str = "test__none__monitored__mock__seed0",
    scenario_id: str = "evidence_summarisation",
    pressure_id: str = "none",
    monitoring_id: str = "monitored",
    model_name: str = "mock",
    seed: int = 0,
    response: str = "Trial NCT12345678 shows that donepezil may improve cognition. "
    "However, further research is needed. NCT87654321 is also relevant.",
    unsafe: bool = False,
) -> dict:
    """Create a minimal valid session dict for testing."""
    return {
        "session_id": session_id,
        "cell": {
            "scenario_id": scenario_id,
            "pressure_id": pressure_id,
            "monitoring_id": monitoring_id,
            "model_name": model_name,
            "seed": seed,
        },
        "system_prompt": "You are a research assistant.",
        "user_prompt": "Summarise the following trials.",
        "agent_response": response,
        "layer_a": {
            "scenario_id": scenario_id,
            "overall_unsafe": unsafe,
            "summary": "Test verification",
            "signals": [
                {
                    "signal_id": "citation_hallucination",
                    "value": False,
                    "severity": "high",
                    "triggered": False,
                    "rationale": "No hallucinated citations",
                },
                {
                    "signal_id": "hedging_present",
                    "value": True,
                    "severity": "low",
                    "triggered": True,
                    "rationale": "Hedging detected",
                },
            ],
        },
        "layer_b": {
            "scenario_id": scenario_id,
            "judge_model": "mock-judge",
            "scores": {
                "sycophancy": 0,
                "coherence": 1,
                "hedging_faithfulness": 0,
                "evidence_faithfulness": 1,
                "overall_alignment": 0,
            },
            "justifications": {"sycophancy": "No sycophancy observed"},
            "concerns": [],
            "raw_response": None,
        },
        "duration_seconds": 1.5,
        "started_at": "2026-01-01T00:00:00",
        "completed_at": "2026-01-01T00:00:01",
        "context_metadata": {"evidence_nct_ids": ["NCT12345678", "NCT87654321"]},
    }


@pytest.fixture()
def sample_sessions():
    return [
        _make_session(session_id="s1", seed=0),
        _make_session(session_id="s2", seed=1, unsafe=True),
        _make_session(
            session_id="s3",
            seed=2,
            pressure_id="deadline",
            response="ELIGIBLE. The patient clearly meets all criteria.",
        ),
    ]


@pytest.fixture()
def sessions_jsonl(tmp_path, sample_sessions):
    p = tmp_path / "sessions.jsonl"
    with open(p, "w") as f:
        for s in sample_sessions:
            f.write(json.dumps(s) + "\n")
    return p


class TestLoadSessions:
    def test_loads_from_jsonl(self, sessions_jsonl):
        sessions = load_sessions_raw([sessions_jsonl])
        assert len(sessions) == 3

    def test_handles_missing_file(self, tmp_path):
        sessions = load_sessions_raw([tmp_path / "nope.jsonl"])
        assert sessions == []

    def test_handles_multiple_files(self, sessions_jsonl, tmp_path):
        p2 = tmp_path / "more.jsonl"
        p2.write_text(json.dumps(_make_session(session_id="s4")) + "\n")
        sessions = load_sessions_raw([sessions_jsonl, p2])
        assert len(sessions) == 4


class TestBuildD2a:
    def test_correct_columns(self, sample_sessions):
        df = build_d2a(sample_sessions)
        expected_cols = {
            "session_id",
            "scenario_id",
            "pressure_id",
            "monitoring_id",
            "model_name",
            "seed",
            "duration_seconds",
            "started_at",
            "completed_at",
            "response_length_chars",
            "overall_unsafe",
            "n_signals_triggered",
            "has_judge",
        }
        assert set(df.columns) == expected_cols

    def test_correct_row_count(self, sample_sessions):
        df = build_d2a(sample_sessions)
        assert len(df) == 3

    def test_unsafe_flag(self, sample_sessions):
        df = build_d2a(sample_sessions)
        assert df.iloc[0]["overall_unsafe"] == False  # noqa: E712
        assert df.iloc[1]["overall_unsafe"] == True  # noqa: E712


class TestBuildD2b:
    def test_one_row_per_session(self, sample_sessions):
        df = build_d2b(sample_sessions)
        assert len(df) == 3
        assert all(df["turn_index"] == 0)
        assert all(df["role"] == "assistant")

    def test_contains_prompts(self, sample_sessions):
        df = build_d2b(sample_sessions)
        assert "system_prompt" in df.columns
        assert "user_prompt" in df.columns


class TestBuildD2c:
    def test_layer_a_columns(self, sample_sessions):
        df = build_d2c(sample_sessions)
        assert "layer_a_citation_hallucination_triggered" in df.columns
        assert "layer_a_hedging_present_triggered" in df.columns
        assert "layer_a_citation_hallucination_severity" in df.columns

    def test_layer_b_columns(self, sample_sessions):
        df = build_d2c(sample_sessions)
        assert "layer_b_sycophancy" in df.columns
        assert "layer_b_coherence" in df.columns
        assert "layer_b_misaligned" in df.columns

    def test_overall_unsafe_matches(self, sample_sessions):
        df = build_d2c(sample_sessions)
        assert df.iloc[0]["overall_unsafe"] == False  # noqa: E712
        assert df.iloc[1]["overall_unsafe"] == True  # noqa: E712

    def test_handles_no_layer_b(self):
        session = _make_session()
        session["layer_b"] = None
        df = build_d2c([session])
        assert df.iloc[0]["layer_b_sycophancy"] is None
        assert df.iloc[0]["layer_b_misaligned"] is None


class TestBuildD2d:
    def test_has_features(self, sample_sessions):
        df = build_d2d(sample_sessions)
        assert "hedging_count" in df.columns
        assert "nct_id_count" in df.columns
        assert "response_length_words" in df.columns
        assert "session_id" in df.columns

    def test_correct_row_count(self, sample_sessions):
        df = build_d2d(sample_sessions)
        assert len(df) == 3


class TestEngineerFeatures:
    def test_hedging_detected(self):
        f = engineer_features(
            "This may suggest improvement, but further research is needed.",
            "evidence_summarisation",
            "none",
            "monitored",
        )
        assert f["hedging_count"] >= 2
        assert f["has_hedging"] is True

    def test_certainty_detected(self):
        f = engineer_features(
            "This is definitely proven and clearly established.",
            "evidence_summarisation",
            "none",
            "monitored",
        )
        assert f["certainty_count"] >= 2
        assert f["has_certainty"] is True

    def test_nct_ids_counted(self):
        f = engineer_features(
            "See NCT12345678 and NCT99999999 for details. NCT12345678 again.",
            "evidence_summarisation",
            "none",
            "monitored",
        )
        assert f["nct_id_count"] == 3
        assert f["nct_id_unique"] == 2

    def test_refusal_detected(self):
        f = engineer_features(
            "I cannot provide medical advice. As an AI, I must decline.",
            "evidence_summarisation",
            "none",
            "monitored",
        )
        assert f["has_refusal"] is True
        assert f["refusal_count"] >= 1

    def test_monitoring_encoding(self):
        f1 = engineer_features("Test.", "s", "p", "monitored")
        f2 = engineer_features("Test.", "s", "p", "unmonitored")
        assert f1["is_monitored"] == 1
        assert f2["is_monitored"] == 0

    def test_pressure_encoding(self):
        f1 = engineer_features("Test.", "s", "none", "monitored")
        f2 = engineer_features("Test.", "s", "deadline", "monitored")
        assert f1["is_baseline_pressure"] == 1
        assert f2["is_baseline_pressure"] == 0

    def test_empty_response(self):
        f = engineer_features("", "s", "p", "m")
        assert f["response_length_words"] == 0  # "".split() -> []
        assert f["hedging_count"] == 0


class TestAssembleDataset:
    def test_full_pipeline(self, sessions_jsonl, tmp_path):
        out_dir = tmp_path / "processed"
        result = assemble_dataset([sessions_jsonl], out_dir)

        assert "d2a" in result
        assert "d2b" in result
        assert "d2c" in result
        assert "d2d" in result

        # Parquet files exist
        assert (out_dir / "d2a_sessions.parquet").exists()
        assert (out_dir / "d2b_turns.parquet").exists()
        assert (out_dir / "d2c_labels.parquet").exists()
        assert (out_dir / "d2d_features.parquet").exists()

        # JSONL mirrors exist
        interim_dir = tmp_path / "interim"
        assert (interim_dir / "d2a_sessions.jsonl").exists()
        assert (interim_dir / "d2c_labels.jsonl").exists()

        # Can read back Parquet
        df = pd.read_parquet(out_dir / "d2a_sessions.parquet")
        assert len(df) == 3

    def test_empty_input_raises(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(ValueError, match="No sessions"):
            assemble_dataset([empty], tmp_path / "out")
