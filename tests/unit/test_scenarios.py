"""Tests for the scenarios module — schema, registry, and prompt rendering."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from neuroguard.scenarios.registry import load_all_scenarios
from neuroguard.scenarios.schema import EvidenceSelection, Scenario

SCENARIOS_DIR = Path("configs/scenarios")


# ---- Schema tests ----


class TestScenarioSchema:
    def test_valid_scenario(self):
        """A fully-specified scenario validates correctly."""
        s = Scenario(
            id="test_scenario",
            name="Test",
            description="A test scenario",
            task_type="summarisation",
            system_prompt="You are a helpful assistant.",
            user_prompt_template="Summarise {intervention} for {condition}.",
            expected_behaviours={"accuracy": "Cites real trials"},
            misalignment_signals={"overclaiming": "Exaggerates results"},
            evidence_selection=EvidenceSelection(strategy="random_subset", n_trials=5),
            template_variables=["intervention", "condition"],
        )
        assert s.id == "test_scenario"
        assert s.task_type == "summarisation"

    def test_missing_required_field(self):
        """Scenario rejects missing required fields."""
        with pytest.raises(ValidationError):
            Scenario(id="incomplete")

    def test_render_user_prompt(self):
        """Template rendering fills placeholders correctly."""
        s = Scenario(
            id="render_test",
            name="Render Test",
            description="Tests rendering",
            task_type="summarisation",
            system_prompt="System.",
            user_prompt_template="Tell me about {drug} for {disease}.",
            expected_behaviours={"a": "b"},
            misalignment_signals={"c": "d"},
            evidence_selection=EvidenceSelection(strategy="random_subset"),
        )
        rendered = s.render_user_prompt(drug="Aducanumab", disease="Alzheimer's")
        assert "Aducanumab" in rendered
        assert "Alzheimer's" in rendered

    def test_render_missing_variable_raises(self):
        """Rendering with missing variables raises KeyError."""
        s = Scenario(
            id="missing_var",
            name="Missing Var",
            description="Tests missing var",
            task_type="triage",
            system_prompt="System.",
            user_prompt_template="Patient {name} age {age}.",
            expected_behaviours={"a": "b"},
            misalignment_signals={"c": "d"},
            evidence_selection=EvidenceSelection(strategy="single_trial"),
        )
        with pytest.raises(KeyError):
            s.render_user_prompt(name="Alice")  # missing 'age'

    def test_evidence_selection_defaults(self):
        """EvidenceSelection has sensible defaults."""
        es = EvidenceSelection(strategy="random_subset")
        assert es.n_trials == 10
        assert es.must_include_phases == []
        assert es.must_have_results_count == 0


# ---- Registry tests ----


class TestRegistry:
    def test_load_all_scenarios(self):
        """All YAML files in configs/scenarios/ load and validate."""
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        assert len(scenarios) == 4
        assert "evidence_summarisation" in scenarios
        assert "trial_ranking" in scenarios
        assert "eligibility_triage" in scenarios
        assert "proposal_critique" in scenarios

    def test_each_scenario_has_required_fields(self):
        """Every loaded scenario has non-empty prompts and signal maps."""
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        for sid, scenario in scenarios.items():
            assert scenario.system_prompt.strip(), f"{sid} has empty system_prompt"
            assert scenario.user_prompt_template.strip(), f"{sid} has empty user_prompt"
            assert (
                len(scenario.expected_behaviours) >= 3
            ), f"{sid} has <3 expected behaviours"
            assert (
                len(scenario.misalignment_signals) >= 3
            ), f"{sid} has <3 misalignment signals"

    def test_scenario_ids_match_filenames(self):
        """Scenario ID matches the YAML filename (enforces naming convention)."""
        for yaml_path in sorted(SCENARIOS_DIR.glob("*.yaml")):
            with open(yaml_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            expected_id = yaml_path.stem
            assert (
                raw["id"] == expected_id
            ), f"File {yaml_path.name} has id='{raw['id']}', expected '{expected_id}'"

    def test_task_types_are_valid(self):
        """All scenarios use one of the four defined task types."""
        valid_types = {"summarisation", "ranking", "triage", "critique"}
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        for sid, scenario in scenarios.items():
            assert (
                scenario.task_type in valid_types
            ), f"{sid} has invalid task_type '{scenario.task_type}'"

    def test_template_variables_present_in_template(self):
        """Declared template_variables actually appear in the prompt template."""
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        for sid, scenario in scenarios.items():
            for var in scenario.template_variables:
                placeholder = "{" + var + "}"
                assert (
                    placeholder in scenario.user_prompt_template
                ), f"{sid} declares variable '{var}' but template lacks '{placeholder}'"

    def test_nonexistent_directory_raises(self, tmp_path):
        """load_all_scenarios raises FileNotFoundError for bad path."""
        with pytest.raises(FileNotFoundError):
            load_all_scenarios(tmp_path / "nonexistent")
