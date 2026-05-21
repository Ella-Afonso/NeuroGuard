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


if __name__ == "__main__":
    app()
