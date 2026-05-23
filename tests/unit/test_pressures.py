"""Tests for the pressures module — schema, registry, and application."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from neuroguard.pressures.apply import apply_pressure
from neuroguard.pressures.registry import load_all_pressures
from neuroguard.pressures.schema import Pressure
from neuroguard.scenarios.registry import load_all_scenarios
from neuroguard.scenarios.schema import EvidenceSelection, Scenario

PRESSURES_DIR = Path("configs/pressures")
SCENARIOS_DIR = Path("configs/scenarios")

EXPECTED_PRESSURE_IDS = {
    "none",
    "deadline",
    "authority",
    "peer_disagreement",
    "sunk_cost",
    "emotional_appeal",
    "hidden_instruction",
}


# ---- Schema tests ----


class TestPressureSchema:
    def test_minimal_valid_pressure(self):
        """A pressure with only required fields validates."""
        p = Pressure(
            id="test",
            name="Test",
            description="A test pressure",
            pressure_type="none",
            severity="low",
            safety_review="Test pressure for unit testing.",
        )
        assert p.id == "test"
        assert p.system_prompt_addition == ""
        assert p.expected_misalignment_signals == []

    def test_invalid_pressure_type_rejected(self):
        """pressure_type must be from the literal enum."""
        with pytest.raises(ValidationError):
            Pressure(
                id="x",
                name="X",
                description="x",
                pressure_type="not_a_type",
                severity="low",
                safety_review="x",
            )

    def test_invalid_severity_rejected(self):
        """severity must be one of low/medium/high."""
        with pytest.raises(ValidationError):
            Pressure(
                id="x",
                name="X",
                description="x",
                pressure_type="none",
                severity="extreme",
                safety_review="x",
            )


# ---- Registry tests ----


class TestRegistry:
    def test_load_all_pressures(self):
        """All 7 pressure YAML files load and validate."""
        pressures = load_all_pressures(PRESSURES_DIR)
        assert set(pressures.keys()) == EXPECTED_PRESSURE_IDS

    def test_pressure_ids_match_filenames(self):
        """Pressure ID matches the YAML filename."""
        for yaml_path in sorted(PRESSURES_DIR.glob("*.yaml")):
            with open(yaml_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            expected_id = yaml_path.stem
            assert (
                raw["id"] == expected_id
            ), f"File {yaml_path.name} has id='{raw['id']}', expected '{expected_id}'"

    def test_every_pressure_has_safety_review(self):
        """Every pressure must document an ethical justification."""
        pressures = load_all_pressures(PRESSURES_DIR)
        for pid, pressure in pressures.items():
            assert pressure.safety_review.strip(), f"{pid} has empty safety_review"
            assert (
                len(pressure.safety_review) >= 50
            ), f"{pid} safety_review too short to be meaningful"

    def test_none_pressure_is_truly_inert(self):
        """The 'none' baseline must not modify any prompts."""
        pressures = load_all_pressures(PRESSURES_DIR)
        none = pressures["none"]
        assert none.system_prompt_addition == ""
        assert none.user_prompt_prefix == ""
        assert none.user_prompt_suffix == ""
        assert none.expected_misalignment_signals == []

    def test_active_pressures_modify_prompts(self):
        """Every non-baseline pressure must modify at least one prompt field."""
        pressures = load_all_pressures(PRESSURES_DIR)
        for pid, pressure in pressures.items():
            if pid == "none":
                continue
            modifies_something = bool(
                pressure.system_prompt_addition
                or pressure.user_prompt_prefix
                or pressure.user_prompt_suffix
            )
            assert modifies_something, f"{pid} is non-baseline but modifies nothing"

    def test_active_pressures_have_hypotheses(self):
        """Every non-baseline pressure must pre-register expected signals."""
        pressures = load_all_pressures(PRESSURES_DIR)
        for pid, pressure in pressures.items():
            if pid == "none":
                continue
            assert (
                pressure.expected_misalignment_signals
            ), f"{pid} has no pre-registered expected signals"


# ---- Apply tests ----


def _make_test_scenario() -> Scenario:
    """Build a minimal scenario for application tests."""
    return Scenario(
        id="t",
        name="T",
        description="t",
        task_type="summarisation",
        system_prompt="You are an aligned assistant.",
        user_prompt_template="Please summarise {topic}.",
        expected_behaviours={"a": "b"},
        misalignment_signals={"c": "d"},
        evidence_selection=EvidenceSelection(strategy="random_subset"),
        template_variables=["topic"],
    )


class TestApply:
    def test_none_pressure_leaves_prompts_unchanged(self):
        """Applying 'none' returns the scenario's prompts verbatim."""
        scenario = _make_test_scenario()
        pressures = load_all_pressures(PRESSURES_DIR)
        system, user = apply_pressure(scenario, pressures["none"])
        assert system == scenario.system_prompt
        assert user == scenario.user_prompt_template

    def test_deadline_adds_prefix(self):
        """Deadline pressure prepends urgency text to the user prompt."""
        scenario = _make_test_scenario()
        pressures = load_all_pressures(PRESSURES_DIR)
        _, user = apply_pressure(scenario, pressures["deadline"])
        assert "URGENT" in user or "urgent" in user.lower()
        assert "{topic}" in user  # placeholder preserved

    def test_hidden_instruction_modifies_system(self):
        """Hidden instruction injects into the system prompt."""
        scenario = _make_test_scenario()
        pressures = load_all_pressures(PRESSURES_DIR)
        system, _ = apply_pressure(scenario, pressures["hidden_instruction"])
        assert system != scenario.system_prompt
        assert len(system) > len(scenario.system_prompt)

    def test_template_variables_preserved(self):
        """All pressures preserve template placeholders in the user prompt."""
        scenario = _make_test_scenario()
        pressures = load_all_pressures(PRESSURES_DIR)
        for pid, pressure in pressures.items():
            _, user = apply_pressure(scenario, pressure)
            assert "{topic}" in user, f"{pid} broke the {{topic}} placeholder"

    def test_factorial_compatibility(self):
        """Every (scenario, pressure) pair produces a usable prompt."""
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        pressures = load_all_pressures(PRESSURES_DIR)
        for sid, scenario in scenarios.items():
            for pid, pressure in pressures.items():
                system, user = apply_pressure(scenario, pressure)
                assert system.strip(), f"({sid}, {pid}) produced empty system prompt"
                assert user.strip(), f"({sid}, {pid}) produced empty user prompt"
                # All scenario placeholders still present
                for var in scenario.template_variables:
                    placeholder = "{" + var + "}"
                    assert (
                        placeholder in user
                    ), f"({sid}, {pid}) lost placeholder {placeholder}"
