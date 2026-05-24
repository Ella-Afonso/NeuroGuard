"""Session orchestrator — runs one or many factorial cells end-to-end."""

from datetime import datetime, timezone
from pathlib import Path

from neuroguard.evidence.schema import TrialRecord
from neuroguard.judge.adapters import JudgeAdapter
from neuroguard.judge.judge import judge_response
from neuroguard.logging_setup import get_logger
from neuroguard.monitoring.apply import compose_prompts
from neuroguard.monitoring.schema import MonitoringCondition
from neuroguard.pressures.schema import Pressure
from neuroguard.runner.adapters import ModelAdapter
from neuroguard.runner.context import build_task_context
from neuroguard.runner.persistence import write_session
from neuroguard.runner.schema import Session, SessionCell
from neuroguard.scenarios.schema import Scenario
from neuroguard.synthetic.schema import PatientProfile, ResearchProposal
from neuroguard.verification.verify import verify_response

logger = get_logger(__name__)


def run_session(
    *,
    scenario: Scenario,
    pressure: Pressure,
    monitoring: MonitoringCondition,
    seed: int,
    model_adapter: ModelAdapter,
    judge_adapter: JudgeAdapter | None = None,
    trial_pool: list[TrialRecord] | None = None,
    patient_pool: list[PatientProfile] | None = None,
    proposal_pool: list[ResearchProposal] | None = None,
) -> Session:
    """Run one factorial cell end-to-end.

    Pipeline:
      1. Build task context (seeded evidence/patient/proposal selection).
      2. Compose system + user prompts (scenario + pressure + monitoring).
      3. Call the model under test.
      4. Run Layer A programmatic verification.
      5. Optionally run Layer B LLM judge.
      6. Bundle everything into a Session.

    Args:
        scenario: Scenario object.
        pressure: Pressure condition.
        monitoring: Monitoring condition.
        seed: Seed for deterministic context selection.
        model_adapter: Adapter for the model under test.
        judge_adapter: Optional adapter for Layer B judge.
        trial_pool: D1 trial pool (required by all current scenarios).
        patient_pool: Patient profile pool (required for triage).
        proposal_pool: Research proposal pool (required for critique).

    Returns:
        A complete Session with both verification layers attached.
    """
    started = datetime.now(timezone.utc)

    context = build_task_context(
        scenario,
        seed,
        trial_pool=trial_pool,
        patient_pool=patient_pool,
        proposal_pool=proposal_pool,
    )

    system_prompt, user_template = compose_prompts(scenario, pressure, monitoring)
    user_prompt = user_template.format(**context.variables)

    agent_response = model_adapter.complete(system_prompt, user_prompt)

    layer_a = verify_response(
        agent_response,
        scenario,
        evidence_nct_ids=context.evidence_nct_ids,
        patient=context.patient,
        proposal=context.proposal,
    )

    layer_b = None
    if judge_adapter is not None:
        # Show the judge the rendered task content so it can grade evidence faithfulness.
        task_context_text = "\n".join(
            f"[{k}]\n{v}" for k, v in context.variables.items()
        )
        layer_b = judge_response(
            agent_response,
            scenario,
            adapter=judge_adapter,
            layer_a=layer_a,
            task_context=task_context_text,
        )

    completed = datetime.now(timezone.utc)
    duration = (completed - started).total_seconds()

    cell = SessionCell(
        scenario_id=scenario.id,
        pressure_id=pressure.id,
        monitoring_id=monitoring.id,
        model_name=model_adapter.model_name,
        seed=seed,
    )
    session = Session(
        session_id=Session.make_session_id(cell),
        cell=cell,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        agent_response=agent_response,
        layer_a=layer_a,
        layer_b=layer_b,
        duration_seconds=duration,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        context_metadata=context.metadata,
    )

    logger.info(
        "session_complete",
        session_id=session.session_id,
        overall_unsafe=layer_a.overall_unsafe,
        triggered_signals=layer_a.triggered_signal_ids(),
        duration_seconds=duration,
    )
    return session


def run_batch(
    *,
    scenarios: list[Scenario],
    pressures: list[Pressure],
    monitorings: list[MonitoringCondition],
    seeds: list[int],
    model_adapter: ModelAdapter,
    judge_adapter: JudgeAdapter | None = None,
    trial_pool: list[TrialRecord] | None = None,
    patient_pool: list[PatientProfile] | None = None,
    proposal_pool: list[ResearchProposal] | None = None,
    output_path: Path | None = None,
) -> list[Session]:
    """Run a full factorial sweep over the supplied lists.

    Cells are iterated in (scenario, pressure, monitoring, seed) nested
    order — outermost = scenario, innermost = seed. If ``output_path`` is
    provided, each completed session is appended to it as JSONL so a
    crashed run loses at most the in-flight session.

    Args:
        scenarios: Scenarios to sweep.
        pressures: Pressure conditions to sweep.
        monitorings: Monitoring conditions to sweep.
        seeds: Seeds to sweep.
        model_adapter: Adapter for the model under test.
        judge_adapter: Optional Layer B judge adapter.
        trial_pool: D1 trial pool.
        patient_pool: Patient profile pool.
        proposal_pool: Research proposal pool.
        output_path: Optional path to append sessions as JSONL.

    Returns:
        List of all completed Sessions in iteration order.
    """
    total_cells = len(scenarios) * len(pressures) * len(monitorings) * len(seeds)
    logger.info(
        "batch_start",
        total_cells=total_cells,
        scenarios=[s.id for s in scenarios],
        pressures=[p.id for p in pressures],
        monitorings=[m.id for m in monitorings],
        seeds=seeds,
        model=model_adapter.model_name,
    )

    sessions: list[Session] = []
    for scenario in scenarios:
        for pressure in pressures:
            for monitoring in monitorings:
                for seed in seeds:
                    session = run_session(
                        scenario=scenario,
                        pressure=pressure,
                        monitoring=monitoring,
                        seed=seed,
                        model_adapter=model_adapter,
                        judge_adapter=judge_adapter,
                        trial_pool=trial_pool,
                        patient_pool=patient_pool,
                        proposal_pool=proposal_pool,
                    )
                    sessions.append(session)
                    if output_path is not None:
                        write_session(session, output_path)

    logger.info("batch_complete", total_sessions=len(sessions))
    return sessions
