"""Smoke tests — verify the package installs and core modules import correctly."""


def test_version():
    """Package version is accessible and matches pyproject.toml."""
    from neuroguard import __version__

    assert __version__ == "0.1.0"


def test_config_loads():
    """Settings object loads with sane defaults (no .env required)."""
    from neuroguard.config import settings

    assert settings.log_level == "INFO"
    assert settings.max_concurrent == 5
    assert settings.cost_limit_usd == 10.0


def test_logger_creates():
    """Structured logger can be created without error."""
    from neuroguard.logging_setup import get_logger

    logger = get_logger("test")
    assert logger is not None


def test_cli_app_exists():
    """Typer app is importable (entry point will resolve)."""
    from neuroguard.cli import app

    assert app is not None
