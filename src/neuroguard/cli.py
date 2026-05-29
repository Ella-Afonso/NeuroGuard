"""NeuroGuard CLI — experiment runner and utilities.

Entry point registered in pyproject.toml as `neuroguard`.
Subcommands are added by later steps (fetch-evidence, run-experiment, etc.).
"""

import typer
from dotenv import load_dotenv

load_dotenv()  # load .env into os.environ before any command runs

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


@app.command("assemble-dataset")
def assemble_dataset_cmd(
    input_path: str = typer.Option(
        "data/sessions/pre_pilot_001.jsonl",
        help="Path to session JSONL file(s). Comma-separated for multiple.",
    ),
    output_dir: str = typer.Option(
        "data/processed",
        help="Directory to write Parquet output files.",
    ),
) -> None:
    """Assemble raw session JSONL into structured Parquet datasets (D2a-D2d).

    Reads one or more session JSONL files, computes features, and writes:
    - D2a (sessions), D2b (turns), D2c (labels), D2d (features) as Parquet
    - JSONL mirrors in data/interim/ for human inspection
    """
    from pathlib import Path

    from neuroguard.dataset import assemble_dataset

    paths = [Path(p.strip()) for p in input_path.split(",")]
    out = Path(output_dir)

    result = assemble_dataset(input_paths=paths, output_dir=out)

    typer.echo("\nDataset assembled:")
    typer.echo(f"  D2a sessions: {len(result['d2a'])} rows")
    typer.echo(f"  D2b turns:    {len(result['d2b'])} rows")
    typer.echo(f"  D2c labels:   {len(result['d2c'])} rows")
    typer.echo(
        f"  D2d features: {len(result['d2d'])} rows ({len(result['d2d'].columns) - 1} features)"
    )
    typer.echo(f"\nParquet files: {out}/")
    typer.echo(f"JSONL mirrors: {out.parent / 'interim'}/")


@app.command("run-batch")
def run_batch_cmd(
    config: str = typer.Option(
        ...,
        help="Path to experiment config YAML (e.g. configs/experiments/pre_pilot.yaml).",
    ),
    dry_run: bool = typer.Option(
        False, help="Print the plan (total cells, cost estimate) without running."
    ),
) -> None:
    """Run a full factorial experiment sweep from a YAML config file.

    Iterates over all (scenario x pressure x monitoring x model x seed) cells
    defined in the config. Sessions are streamed to the configured output JSONL
    so a crashed run loses at most the in-flight session.
    """
    import json
    import time
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
        write_session,
    )
    from neuroguard.runner.config import load_experiment_config
    from neuroguard.scenarios import get_scenario
    from neuroguard.synthetic import load_patient_profiles, load_research_proposals

    cfg = load_experiment_config(Path(config))
    typer.echo(f"Experiment: {cfg.name}")
    typer.echo(f"Total cells: {cfg.total_cells()}")
    typer.echo(f"Cost estimate: ${cfg.cost_estimate_usd():.2f}")
    typer.echo(f"Output: {cfg.output_path}")

    if dry_run:
        typer.echo("\n[dry-run] No sessions will be executed.")
        return

    # Resolve shared resources
    trial_pool = load_trial_pool(Path(cfg.trials_path))
    patient_pool = load_patient_profiles()
    proposal_pool = load_research_proposals()

    # Resolve scenarios, pressures, monitoring
    scenarios = [get_scenario(sid) for sid in cfg.scenarios]
    pressures = [get_pressure(pid) for pid in cfg.pressures]
    monitorings = [get_monitoring(mid) for mid in cfg.monitoring]

    # Build judge adapter (shared across all cells)
    judge_adapter = None
    if cfg.judge.provider == "mock":
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
            ),
            model_name=cfg.judge.name or "mock-judge",
        )
    elif cfg.judge.provider == "openai":
        judge_adapter = OpenAIJudgeAdapter(
            model_name=cfg.judge.name,
            temperature=cfg.judge.temperature,
            max_tokens=cfg.judge.max_tokens,
        )

    # Run factorial
    out = Path(cfg.output_path)
    completed = 0
    total = cfg.total_cells()
    t_start = time.time()

    for model_spec in cfg.models:
        # Build model adapter for this model
        if model_spec.provider == "mock":
            model_adapter = MockModelAdapter(
                "(mock model response)", model_name=model_spec.name
            )
        elif model_spec.provider == "openai":
            model_adapter = OpenAIModelAdapter(
                model_name=model_spec.name,
                temperature=model_spec.temperature,
                max_tokens=model_spec.max_tokens,
            )
        elif model_spec.provider == "anthropic":
            model_adapter = AnthropicModelAdapter(
                model_name=model_spec.name,
                temperature=model_spec.temperature,
                max_tokens=model_spec.max_tokens,
            )
        else:
            raise typer.BadParameter(f"Unknown model provider: {model_spec.provider}")

        for scenario in scenarios:
            for pressure in pressures:
                for monitoring_cond in monitorings:
                    for seed in cfg.seeds:
                        session = run_session(
                            scenario=scenario,
                            pressure=pressure,
                            monitoring=monitoring_cond,
                            seed=seed,
                            model_adapter=model_adapter,
                            judge_adapter=judge_adapter,
                            trial_pool=trial_pool,
                            patient_pool=patient_pool,
                            proposal_pool=proposal_pool,
                        )
                        write_session(session, out)
                        completed += 1
                        elapsed = time.time() - t_start
                        rate = completed / elapsed if elapsed > 0 else 0
                        typer.echo(
                            f"[{completed}/{total}] {session.session_id} "
                            f"unsafe={session.layer_a.overall_unsafe} "
                            f"({rate:.1f} sess/s)"
                        )

    elapsed_total = time.time() - t_start
    typer.echo(f"\nBatch complete: {completed} sessions in {elapsed_total:.1f}s")
    typer.echo(f"Output: {out}")


if __name__ == "__main__":
    app()
