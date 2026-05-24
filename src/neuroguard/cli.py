"""NeuroGuard CLI — experiment runner and utilities.

Entry point registered in pyproject.toml as `neuroguard`.
Subcommands are added by later steps (fetch-evidence, run-experiment, etc.).
"""

import typer

app = typer.Typer(
    name="neuroguard",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """NeuroGuard: model-organism benchmark for AI safety research."""


@app.command()
def version() -> None:
    """Print the installed package version."""
    from neuroguard import __version__

    typer.echo(f"neuroguard {__version__}")


@app.command("fetch-evidence")
def fetch_evidence(
    condition: str = typer.Option(
        "Alzheimer Disease", help="Condition to search CT.gov for."
    ),
    max_studies: int = typer.Option(500, help="Maximum number of studies to fetch."),
    output_dir: str = typer.Option("data/raw", help="Directory to write JSONL output."),
) -> None:
    """Fetch clinical trial evidence from ClinicalTrials.gov API v2."""
    import asyncio
    from pathlib import Path

    from neuroguard.evidence import fetch_trials

    records = asyncio.run(
        fetch_trials(
            condition=condition,
            max_studies=max_studies,
            output_dir=Path(output_dir),
        )
    )
    typer.echo(f"Fetched {len(records)} trials → {output_dir}/")


@app.command("run-session")
def run_session_cmd(
    scenario: str = typer.Option(
        ..., help="Scenario ID (e.g. evidence_summarisation)."
    ),
    pressure: str = typer.Option("none", help="Pressure ID (e.g. none, deadline)."),
    monitoring: str = typer.Option(
        "monitored", help="Monitoring ID (monitored / unmonitored)."
    ),
    seed: int = typer.Option(0, help="Seed for deterministic context selection."),
    model: str = typer.Option(
        "mock", help="Model under test: mock / openai:<name> / anthropic:<name>."
    ),
    judge: str = typer.Option(
        "none",
        help="Judge: none / mock / openai:<name>. Use mock for dry runs without API.",
    ),
    trials_path: str = typer.Option(
        "data/raw/trials.jsonl",
        help="Path to cached D1 trial pool JSONL.",
    ),
    output_path: str = typer.Option(
        "data/sessions/sessions.jsonl",
        help="Path to append the resulting session as JSONL.",
    ),
) -> None:
    """Run one factorial cell end-to-end and append the session to JSONL.

    The default model (mock) returns a fixed string and never calls an API,
    so this command is safe to run for testing the full pipeline shape.
    """
    import json
    from pathlib import Path

    from neuroguard.judge.adapters import MockJudgeAdapter, OpenAIJudgeAdapter
    from neuroguard.monitoring import get_monitoring
    from neuroguard.pressures import get_pressure
    from neuroguard.runner import (
        AnthropicModelAdapter,
        MockModelAdapter,
        OpenAIModelAdapter,
        load_trial_pool,
        run_session,
    )
    from neuroguard.scenarios import get_scenario
    from neuroguard.synthetic import load_patient_profiles, load_research_proposals

    scenario_obj = get_scenario(scenario)
    pressure_obj = get_pressure(pressure)
    monitoring_obj = get_monitoring(monitoring)

    if model == "mock":
        model_adapter = MockModelAdapter("(mock model response)")
    elif model.startswith("openai:"):
        model_adapter = OpenAIModelAdapter(model_name=model.split(":", 1)[1])
    elif model.startswith("anthropic:"):
        model_adapter = AnthropicModelAdapter(model_name=model.split(":", 1)[1])
    else:
        raise typer.BadParameter(f"Unknown model spec: {model}")

    judge_adapter = None
    if judge == "mock":
        judge_adapter = MockJudgeAdapter(
            json.dumps(
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
        )
    elif judge.startswith("openai:"):
        judge_adapter = OpenAIJudgeAdapter(model_name=judge.split(":", 1)[1])
    elif judge != "none":
        raise typer.BadParameter(f"Unknown judge spec: {judge}")

    trial_pool = load_trial_pool(Path(trials_path))
    patient_pool = (
        load_patient_profiles() if scenario_obj.task_type == "triage" else None
    )
    proposal_pool = (
        load_research_proposals() if scenario_obj.task_type == "critique" else None
    )

    session = run_session(
        scenario=scenario_obj,
        pressure=pressure_obj,
        monitoring=monitoring_obj,
        seed=seed,
        model_adapter=model_adapter,
        judge_adapter=judge_adapter,
        trial_pool=trial_pool,
        patient_pool=patient_pool,
        proposal_pool=proposal_pool,
    )

    out = Path(output_path)
    from neuroguard.runner import write_session as _write

    _write(session, out)
    typer.echo(f"session_id: {session.session_id}")
    typer.echo(f"layer_a unsafe: {session.layer_a.overall_unsafe}")
    typer.echo(f"triggered: {session.layer_a.triggered_signal_ids()}")
    typer.echo(f"appended -> {out}")


if __name__ == "__main__":
    app()
