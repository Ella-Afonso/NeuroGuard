"""Tests for experiment config loading and batch CLI."""

from pathlib import Path

import pytest

from neuroguard.runner.config import ExperimentConfig, load_experiment_config

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments"


class TestExperimentConfig:
    def test_pre_pilot_loads(self):
        cfg = load_experiment_config(CONFIGS_DIR / "pre_pilot.yaml")
        assert cfg.name == "pre_pilot_001"
        assert cfg.total_cells() == 8
        assert cfg.cost_estimate_usd() == pytest.approx(0.04, abs=0.01)
        assert cfg.scenarios == ["evidence_summarisation"]
        assert cfg.pressures == ["none", "deadline"]
        assert cfg.monitoring == ["monitored", "unmonitored"]
        assert len(cfg.models) == 1
        assert cfg.models[0].provider == "openai"
        assert cfg.models[0].name == "gpt-4o-mini"
        assert cfg.seeds == [0, 1]
        assert cfg.judge.provider == "openai"
        assert cfg.judge.name == "gpt-4o-mini"

    def test_pre_pilot_mock_loads(self):
        cfg = load_experiment_config(CONFIGS_DIR / "pre_pilot_mock.yaml")
        assert cfg.name == "pre_pilot_mock"
        assert cfg.total_cells() == 16
        assert cfg.models[0].provider == "mock"
        assert cfg.judge.provider == "mock"

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_experiment_config(tmp_path / "nope.yaml")

    def test_empty_config_raises(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_experiment_config(empty)

    def test_invalid_config_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: x\n")
        with pytest.raises((ValueError, TypeError)):
            load_experiment_config(bad)

    def test_total_cells_formula(self):
        cfg = ExperimentConfig(
            name="test",
            scenarios=["a", "b"],
            pressures=["p1", "p2", "p3"],
            monitoring=["m1"],
            models=[
                {"provider": "mock", "name": "x"},
                {"provider": "mock", "name": "y"},
            ],
            seeds=[0, 1, 2],
        )
        # 2 x 3 x 1 x 2 x 3 = 36
        assert cfg.total_cells() == 36

    def test_cost_estimate(self):
        cfg = ExperimentConfig(
            name="test",
            scenarios=["a"],
            pressures=["p"],
            monitoring=["m"],
            models=[{"provider": "mock", "name": "x"}],
            seeds=[0],
        )
        assert cfg.cost_estimate_usd(0.01) == pytest.approx(0.01)


class TestRunBatchCLI:
    """Integration tests for the run-batch CLI command with mock config."""

    def test_dry_run(self, capsys):
        from typer.testing import CliRunner

        from neuroguard.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-batch",
                "--config",
                str(CONFIGS_DIR / "pre_pilot_mock.yaml"),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "pre_pilot_mock" in result.output
        assert "Total cells: 16" in result.output
        assert "[dry-run]" in result.output

    def test_mock_batch_runs(self, tmp_path):
        """Run the mock batch config and verify output JSONL is produced."""
        import json

        import yaml

        from neuroguard.runner.persistence import read_sessions

        # Create a minimal mock trials.jsonl so the runner can load evidence
        trials_path = tmp_path / "trials.jsonl"
        mock_trial = {
            "nct_id": "NCT00000001",
            "brief_title": "Mock Alzheimer Trial",
            "conditions": ["Alzheimer Disease"],
            "interventions": ["Donepezil"],
            "phase": "PHASE3",
            "overall_status": "COMPLETED",
            "study_type": "INTERVENTIONAL",
            "enrollment": 100,
            "primary_outcomes": ["ADAS-Cog score change"],
            "secondary_outcomes": [],
            "has_results": True,
            "source_url": "https://clinicaltrials.gov/study/NCT00000001",
            "fetched_at": "2026-01-01T00:00:00Z",
        }
        with open(trials_path, "w") as f:
            f.write(json.dumps(mock_trial) + "\n")

        # Create a tiny config pointing output to tmp
        out_path = tmp_path / "test_batch.jsonl"
        cfg_data = {
            "name": "test_batch",
            "scenarios": ["evidence_summarisation"],
            "pressures": ["none"],
            "monitoring": ["monitored"],
            "models": [{"provider": "mock", "name": "mock-test"}],
            "judge": {"provider": "mock", "name": "mock-judge"},
            "seeds": [0, 1],
            "output_path": str(out_path),
            "trials_path": str(trials_path),
        }
        cfg_path = tmp_path / "cfg.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_data, f)

        from typer.testing import CliRunner

        from neuroguard.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["run-batch", "--config", str(cfg_path)])
        assert result.exit_code == 0, result.output
        assert "Batch complete: 2 sessions" in result.output

        # Verify JSONL contains valid sessions
        sessions = read_sessions(out_path)
        assert len(sessions) == 2
        assert sessions[0].cell.model_name == "mock-test"
        assert sessions[1].cell.seed == 1
