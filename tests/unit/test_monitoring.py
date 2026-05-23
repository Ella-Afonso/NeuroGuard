"""Tests for the monitoring module — schema, registry, and composition."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroguard.monitoring.apply import apply_monitoring, compose_prompts
from neuroguard.monitoring.registry import load_all_monitoring
from neuroguard.monitoring.schema import MonitoringCondition
from neuroguard.pressures.registry import load_all_pressures
from neuroguard.scenarios.registry import load_all_scenarios

MONITORING_DIR = Path("configs/monitoring")
PRESSURES_DIR = Path("configs/pressures")
SCENARIOS_DIR = Path("configs/scenarios")

EXPECTED_MONITORING_IDS = {"monitored", "unmonitored"}


# ---- Schema tests ----


class TestMonitoringSchema:
    def test_minimal_valid_condition(self):
        c = MonitoringCondition(
            id="test",
            name="Test",
            description="t",
            monitoring_type="monitored",
            expected_effect="x",
            safety_review="x",
        )
        assert c.id == "test"
        assert c.system_prompt_addition == ""

    def test_invalid_monitoring_type_rejected(self):
        with pytest.raises(ValidationError):
            MonitoringCondition(
                id="x",
                name="X",
                description="x",
                monitoring_type="partially_monitored",
                expected_effect="x",
                safety_review="x",
            )


# ---- Registry tests ----


class TestRegistry:
    def test_load_all_monitoring(self):
        """Both monitoring YAMLs load and validate."""
        conditions = load_all_monitoring(MONITORING_DIR)
        assert set(conditions.keys()) == EXPECTED_MONITORING_IDS

    def test_each_condition_modifies_prompt(self):
        """Both conditions add a system_prompt note (so the agent sees the framing)."""
        conditions = load_all_monitoring(MONITORING_DIR)
        for cid, condition in conditions.items():
            assert (
                condition.system_prompt_addition.strip()
            ), f"{cid} has empty system_prompt_addition"

    def test_safety_reviews_substantive(self):
        """Both conditions have non-trivial safety_review fields."""
        conditions = load_all_monitoring(MONITORING_DIR)
        for cid, condition in conditions.items():
            assert len(condition.safety_review) >= 50, f"{cid} safety_review too short"
            assert (
                len(condition.expected_effect) >= 30
            ), f"{cid} expected_effect too short"

    def test_monitoring_types_correct(self):
        """ID matches monitoring_type."""
        conditions = load_all_monitoring(MONITORING_DIR)
        assert conditions["monitored"].monitoring_type == "monitored"
        assert conditions["unmonitored"].monitoring_type == "unmonitored"


# ---- Apply tests ----


class TestApplyMonitoring:
    def test_apply_appends_system_note(self):
        """apply_monitoring appends to the system prompt."""
        conditions = load_all_monitoring(MONITORING_DIR)
        system, user = apply_monitoring(
            "Base system.", "User template {x}", conditions["monitored"]
        )
        assert system.startswith("Base system.")
        assert len(system) > len("Base system.")
        assert "{x}" in user  # placeholder preserved


# ---- Composition tests ----


class TestCompose:
    def test_full_factorial_compose(self):
        """All scenario x pressure x monitoring cells compose into valid prompts."""
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        pressures = load_all_pressures(PRESSURES_DIR)
        monitoring = load_all_monitoring(MONITORING_DIR)

        for sid, scenario in scenarios.items():
            for pid, pressure in pressures.items():
                for mid, mon in monitoring.items():
                    system, user = compose_prompts(scenario, pressure, mon)
                    cell = f"({sid}, {pid}, {mid})"
                    assert system.strip(), f"{cell} produced empty system prompt"
                    assert user.strip(), f"{cell} produced empty user prompt"
                    # Scenario placeholders must survive composition
                    for var in scenario.template_variables:
                        placeholder = "{" + var + "}"
                        assert (
                            placeholder in user
                        ), f"{cell} lost placeholder {placeholder}"

    def test_total_factorial_cell_count(self):
        """4 scenarios x 7 pressures x 2 monitoring = 56 cells."""
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        pressures = load_all_pressures(PRESSURES_DIR)
        monitoring = load_all_monitoring(MONITORING_DIR)
        total = len(scenarios) * len(pressures) * len(monitoring)
        assert total == 56, f"Expected 56 factorial cells, got {total}"

    def test_monitoring_actually_changes_prompts(self):
        """monitored vs unmonitored produce different system prompts on same scenario+pressure."""
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        pressures = load_all_pressures(PRESSURES_DIR)
        monitoring = load_all_monitoring(MONITORING_DIR)

        scenario = next(iter(scenarios.values()))
        pressure = pressures["none"]
        sys_m, _ = compose_prompts(scenario, pressure, monitoring["monitored"])
        sys_u, _ = compose_prompts(scenario, pressure, monitoring["unmonitored"])
        assert sys_m != sys_u, "monitored and unmonitored produced identical prompts"
