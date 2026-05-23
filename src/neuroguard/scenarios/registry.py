"""Scenario registry — loads and caches scenarios from YAML configs.

Scenarios are defined in configs/scenarios/*.yaml and loaded into
validated Scenario objects on first access.
"""

from pathlib import Path

import yaml

from neuroguard.logging_setup import get_logger
from neuroguard.scenarios.schema import Scenario

logger = get_logger(__name__)

DEFAULT_SCENARIOS_DIR = Path("configs/scenarios")

_registry: dict[str, Scenario] = {}


def load_all_scenarios(scenarios_dir: Path | None = None) -> dict[str, Scenario]:
    """Load all scenario YAML files from the given directory.

    Args:
        scenarios_dir: Path to directory containing scenario YAML files.
            Defaults to configs/scenarios/.

    Returns:
        Dict mapping scenario_id → validated Scenario object.

    Raises:
        FileNotFoundError: If the scenarios directory doesn't exist.
        ValidationError: If a YAML file fails schema validation.
    """
    global _registry

    if scenarios_dir is None:
        scenarios_dir = DEFAULT_SCENARIOS_DIR

    if not scenarios_dir.exists():
        msg = f"Scenarios directory not found: {scenarios_dir}"
        raise FileNotFoundError(msg)

    _registry.clear()

    for yaml_path in sorted(scenarios_dir.glob("*.yaml")):
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        scenario = Scenario.model_validate(raw)
        _registry[scenario.id] = scenario
        logger.info("scenario_loaded", id=scenario.id, name=scenario.name)

    logger.info("all_scenarios_loaded", count=len(_registry))
    return _registry


def get_scenario(scenario_id: str) -> Scenario:
    """Get a single scenario by ID. Loads registry if empty.

    Args:
        scenario_id: The scenario identifier.

    Returns:
        The validated Scenario object.

    Raises:
        KeyError: If the scenario_id is not found.
    """
    if not _registry:
        load_all_scenarios()

    if scenario_id not in _registry:
        available = list(_registry.keys())
        msg = f"Scenario '{scenario_id}' not found. Available: {available}"
        raise KeyError(msg)

    return _registry[scenario_id]


def list_scenarios() -> list[str]:
    """List all registered scenario IDs.

    Returns:
        Sorted list of scenario identifiers.
    """
    if not _registry:
        load_all_scenarios()
    return sorted(_registry.keys())
