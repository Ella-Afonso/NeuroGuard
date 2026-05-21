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


if __name__ == "__main__":
    app()
