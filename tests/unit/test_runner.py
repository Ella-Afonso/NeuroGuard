"""Tests for the session runner — schema, adapters, context, runner, persistence."""

import json
from datetime import datetime, timezone

import pytest

from neuroguard.evidence.schema import TrialRecord
from neuroguard.judge.adapters import MockJudgeAdapter
from neuroguard.monitoring import get_monitoring
from neuroguard.pressures import get_pressure
from neuroguard.runner import (
    MockModelAdapter,
    Session,
    SessionCell,
    build_task_context,
    read_sessions,
    run_batch,
    run_session,
    select_evidence,
    write_session,
)
from neuroguard.scenarios import get_scenario
from neuroguard.scenarios.schema import EvidenceSelection
from neuroguard.synthetic.loader import load_patient_profiles, load_research_proposals

# ---- Fixtures ----


def _make_trial(
    nct_id: str,
    *,
    phase: str = "PHASE3",
    status: str = "COMPLETED",
    has_results: bool = True,
    interventions: list[str] | None = None,
    conditions: list[str] | None = None,
) -> TrialRecord:
    return TrialRecord(
        nct_id=nct_id,
        brief_title=f"Trial {nct_id}",
        conditions=conditions or ["Alzheimer Disease"],
        interventions=interventions or ["lecanemab"],
        phase=phase,
        overall_status=status,
        enrollment=200,
        primary_outcomes=["CDR-SB Change"],
        has_results=has_results,
        eligibility_criteria="Age 50-85, MMSE 18-26.",
        lead_sponsor="Test Sponsor",
        source_url=f"https://clinicaltrials.gov/study/{nct_id}",
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def trial_pool() -> list[TrialRecord]:
    """Small synthetic trial pool covering multiple phases and statuses."""
    return [
        _make_trial("NCT00000001", phase="PHASE3", status="COMPLETED"),
        _make_trial("NCT00000002", phase="PHASE2", status="COMPLETED"),
        _make_trial(
            "NCT00000003", phase="PHASE3", status="RECRUITING", has_results=False
        ),
        _make_trial(
            "NCT00000004", phase="PHASE1", status="COMPLETED", has_results=False
        ),
        _make_trial("NCT00000005", phase="PHASE3", status="COMPLETED"),
    ]


@pytest.fixture
def aligned_judge_json() -> str:
    return json.dumps(
        {
            "scores": {
                "sycophancy": 0,
                "coherence": 0,
                "hedging_faithfulness": 0,
                "evidence_faithfulness": 0,
                "overall_alignment": 0,
            },
            "justifications": {},
            "concerns": [],
        }
    )


# ---- Adapter tests ----


class TestMockModelAdapter:
    def test_fixed_response(self):
        adapter = MockModelAdapter("hello world")
        assert adapter.complete("sys", "user") == "hello world"
        assert adapter.model_name == "mock-model"

    def test_callable_response(self):
        adapter = MockModelAdapter(lambda s, u: f"got: {u}")
        assert adapter.complete("sys", "ping") == "got: ping"


# ---- Evidence selection tests ----


class TestSelectEvidence:
    def test_random_subset_respects_n_trials(self, trial_pool):
        config = EvidenceSelection(strategy="random_subset", n_trials=3)
        selected = select_evidence(config, trial_pool, seed=42)
        assert len(selected) == 3

    def test_random_subset_deterministic(self, trial_pool):
        config = EvidenceSelection(strategy="random_subset", n_trials=3)
        a = [t.nct_id for t in select_evidence(config, trial_pool, seed=42)]
        b = [t.nct_id for t in select_evidence(config, trial_pool, seed=42)]
        assert a == b

    def test_must_include_phases_satisfied(self, trial_pool):
        config = EvidenceSelection(
            strategy="random_subset",
            n_trials=3,
            must_include_phases=["PHASE2"],
        )
        selected = select_evidence(config, trial_pool, seed=0)
        assert any(t.phase == "PHASE2" for t in selected)

    def test_must_have_results_count_satisfied(self, trial_pool):
        config = EvidenceSelection(
            strategy="random_subset",
            n_trials=3,
            must_have_results_count=2,
        )
        selected = select_evidence(config, trial_pool, seed=0)
        assert sum(1 for t in selected if t.has_results) >= 2

    def test_single_trial(self, trial_pool):
        config = EvidenceSelection(strategy="single_trial")
        selected = select_evidence(config, trial_pool, seed=0)
        assert len(selected) == 1

    def test_empty_pool_raises(self):
        config = EvidenceSelection(strategy="random_subset")
        with pytest.raises(ValueError, match="empty"):
            select_evidence(config, [], seed=0)

    def test_unknown_strategy_raises(self, trial_pool):
        config = EvidenceSelection(strategy="not_a_strategy")
        with pytest.raises(ValueError, match="Unknown"):
            select_evidence(config, trial_pool, seed=0)


# ---- Context tests ----


class TestBuildTaskContext:
    def test_summarisation_context(self, trial_pool):
        scenario = get_scenario("evidence_summarisation")
        context = build_task_context(scenario, seed=0, trial_pool=trial_pool)
        assert {"intervention", "condition", "evidence_block"} <= set(
            context.variables.keys()
        )
        assert context.evidence_nct_ids
        assert all(nct.startswith("NCT") for nct in context.evidence_nct_ids)

    def test_ranking_context(self, trial_pool):
        scenario = get_scenario("trial_ranking")
        context = build_task_context(scenario, seed=0, trial_pool=trial_pool)
        assert "clinical_question" in context.variables
        assert "evidence_block" in context.variables

    def test_triage_context(self, trial_pool):
        scenario = get_scenario("eligibility_triage")
        patients = load_patient_profiles()
        context = build_task_context(
            scenario, seed=0, trial_pool=trial_pool, patient_pool=patients
        )
        assert context.patient is not None
        assert "patient_profile" in context.variables
        assert context.patient.patient_id in context.variables["patient_profile"]

    def test_critique_context(self, trial_pool):
        scenario = get_scenario("proposal_critique")
        proposals = load_research_proposals()
        context = build_task_context(
            scenario,
            seed=0,
            trial_pool=trial_pool,
            proposal_pool=proposals,
        )
        assert context.proposal is not None
        assert "proposal_text" in context.variables

    def test_summarisation_missing_pool_raises(self):
        scenario = get_scenario("evidence_summarisation")
        with pytest.raises(ValueError, match="trial_pool required"):
            build_task_context(scenario, seed=0)

    def test_triage_missing_patient_pool_raises(self, trial_pool):
        scenario = get_scenario("eligibility_triage")
        with pytest.raises(ValueError, match="patient_pool"):
            build_task_context(scenario, seed=0, trial_pool=trial_pool)

    def test_context_deterministic(self, trial_pool):
        scenario = get_scenario("evidence_summarisation")
        a = build_task_context(scenario, seed=42, trial_pool=trial_pool)
        b = build_task_context(scenario, seed=42, trial_pool=trial_pool)
        assert a.evidence_nct_ids == b.evidence_nct_ids


# ---- Runner tests ----


class TestRunSession:
    def test_run_summarisation_session(self, trial_pool):
        session = run_session(
            scenario=get_scenario("evidence_summarisation"),
            pressure=get_pressure("none"),
            monitoring=get_monitoring("monitored"),
            seed=0,
            model_adapter=MockModelAdapter(
                "Preliminary findings; further research needed."
            ),
            trial_pool=trial_pool,
        )
        assert session.cell.scenario_id == "evidence_summarisation"
        assert session.cell.pressure_id == "none"
        assert session.cell.monitoring_id == "monitored"
        assert session.cell.seed == 0
        assert session.cell.model_name == "mock-model"
        assert session.layer_a is not None
        assert session.layer_b is None  # no judge supplied
        assert session.duration_seconds >= 0
        assert session.context_metadata["selected_nct_ids"]

    def test_run_with_judge(self, trial_pool, aligned_judge_json):
        session = run_session(
            scenario=get_scenario("evidence_summarisation"),
            pressure=get_pressure("none"),
            monitoring=get_monitoring("monitored"),
            seed=0,
            model_adapter=MockModelAdapter("Preliminary; further research needed."),
            judge_adapter=MockJudgeAdapter(aligned_judge_json),
            trial_pool=trial_pool,
        )
        assert session.layer_b is not None
        assert session.layer_b.scores.max_score() == 0
        assert session.layer_b.is_misaligned() is False

    def test_run_triage_session(self, trial_pool):
        patients = load_patient_profiles()
        eligible = next(p for p in patients if p.eligibility_truth == "ELIGIBLE")
        # Use a model adapter that returns a correct ELIGIBLE response
        session = run_session(
            scenario=get_scenario("eligibility_triage"),
            pressure=get_pressure("none"),
            monitoring=get_monitoring("monitored"),
            seed=0,
            model_adapter=MockModelAdapter(
                lambda s, u: (
                    "After cautious review of preliminary data, the patient is ELIGIBLE."
                    if "ELIGIBLE" in u or eligible.patient_id in u
                    else "INSUFFICIENT data."
                )
            ),
            trial_pool=trial_pool,
            patient_pool=patients,
        )
        assert session.cell.scenario_id == "eligibility_triage"
        assert session.context_metadata["patient_id"]

    def test_run_critique_session(self, trial_pool):
        proposals = load_research_proposals()
        session = run_session(
            scenario=get_scenario("proposal_critique"),
            pressure=get_pressure("none"),
            monitoring=get_monitoring("monitored"),
            seed=0,
            model_adapter=MockModelAdapter("Concerns about methodology raised."),
            trial_pool=trial_pool,
            proposal_pool=proposals,
        )
        assert session.cell.scenario_id == "proposal_critique"
        assert session.context_metadata["proposal_id"]

    def test_session_id_is_stable_for_same_cell(self, trial_pool):
        kwargs = {
            "scenario": get_scenario("evidence_summarisation"),
            "pressure": get_pressure("none"),
            "monitoring": get_monitoring("monitored"),
            "seed": 7,
            "model_adapter": MockModelAdapter("x"),
            "trial_pool": trial_pool,
        }
        s1 = run_session(**kwargs)
        s2 = run_session(**kwargs)
        assert s1.session_id == s2.session_id

    def test_response_flows_through_layer_a(self, trial_pool):
        # An overclaim response should trigger overclaiming signal
        bad_response = "This drug is definitively shown to cure all patients. No doubt."
        session = run_session(
            scenario=get_scenario("evidence_summarisation"),
            pressure=get_pressure("none"),
            monitoring=get_monitoring("monitored"),
            seed=0,
            model_adapter=MockModelAdapter(bad_response),
            trial_pool=trial_pool,
        )
        assert session.layer_a.overall_unsafe is True
        assert "overclaiming" in session.layer_a.triggered_signal_ids()


# ---- Batch tests ----


class TestRunBatch:
    def test_batch_runs_full_factorial(self, trial_pool, tmp_path):
        scenarios = [get_scenario("evidence_summarisation")]
        pressures = [get_pressure("none"), get_pressure("deadline")]
        monitorings = [get_monitoring("monitored"), get_monitoring("unmonitored")]
        seeds = [0, 1]
        adapter = MockModelAdapter("Preliminary; further research needed.")
        out = tmp_path / "batch.jsonl"

        sessions = run_batch(
            scenarios=scenarios,
            pressures=pressures,
            monitorings=monitorings,
            seeds=seeds,
            model_adapter=adapter,
            trial_pool=trial_pool,
            output_path=out,
        )
        # 1 scenario x 2 pressures x 2 monitoring x 2 seeds = 8 cells
        assert len(sessions) == 8
        # Persisted to JSONL
        assert out.exists()
        loaded = read_sessions(out)
        assert len(loaded) == 8


# ---- Persistence tests ----


class TestPersistence:
    def test_round_trip(self, trial_pool, tmp_path):
        session = run_session(
            scenario=get_scenario("evidence_summarisation"),
            pressure=get_pressure("none"),
            monitoring=get_monitoring("monitored"),
            seed=0,
            model_adapter=MockModelAdapter("Preliminary; more research needed."),
            trial_pool=trial_pool,
        )
        path = tmp_path / "sessions.jsonl"
        write_session(session, path)
        write_session(session, path)
        loaded = read_sessions(path)
        assert len(loaded) == 2
        assert loaded[0].session_id == session.session_id

    def test_read_missing_file_returns_empty(self, tmp_path):
        assert read_sessions(tmp_path / "missing.jsonl") == []

    def test_empty_lines_skipped(self, trial_pool, tmp_path):
        session = run_session(
            scenario=get_scenario("evidence_summarisation"),
            pressure=get_pressure("none"),
            monitoring=get_monitoring("monitored"),
            seed=0,
            model_adapter=MockModelAdapter("ok"),
            trial_pool=trial_pool,
        )
        path = tmp_path / "with_blanks.jsonl"
        write_session(session, path)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n  \n")
        write_session(session, path)
        assert len(read_sessions(path)) == 2


# ---- Schema sanity ----


class TestSessionSchema:
    def test_session_id_format(self):
        cell = SessionCell(
            scenario_id="s",
            pressure_id="p",
            monitoring_id="m",
            model_name="mdl",
            seed=42,
        )
        assert Session.make_session_id(cell) == "s__p__m__mdl__seed42"
