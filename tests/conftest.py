"""Shared pytest fixtures for NeuroGuard tests.

Fixtures defined here are automatically available to all test modules
without explicit import.
"""

import pytest


@pytest.fixture
def sample_nct_id() -> str:
    """A valid NCT ID for testing evidence-layer code."""
    return "NCT00000001"


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure mirroring data/."""
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    for d in (raw, interim, processed):
        d.mkdir()
    return tmp_path


@pytest.fixture
def sample_session_dict() -> dict:
    """A minimal valid session dictionary for schema validation tests."""
    return {
        "session_id": "test-session-001",
        "scenario_id": "evidence_summary",
        "pressure_condition": "none",
        "monitoring_condition": "monitored",
        "task_type": "summarisation",
        "model_provider": "anthropic",
        "model_name": "claude-sonnet-4-20250514",
        "model_version": "20250514",
        "seed": 42,
        "n_turns": 3,
        "total_input_tokens": 1500,
        "total_output_tokens": 800,
        "cost_usd": 0.012,
        "status": "completed",
    }
