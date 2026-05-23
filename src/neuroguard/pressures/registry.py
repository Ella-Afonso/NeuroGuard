"""Pressure registry — loads and caches pressures from YAML configs."""

from pathlib import Path

import yaml

from neuroguard.logging_setup import get_logger
from neuroguard.pressures.schema import Pressure

logger = get_logger(__name__)

DEFAULT_PRESSURES_DIR = Path("configs/pressures")

_registry: dict[str, Pressure] = {}


def load_all_pressures(pressures_dir: Path | None = None) -> dict[str, Pressure]:
    """Load all pressure YAML files from the given directory.

    Args:
        pressures_dir: Directory containing pressure YAML files. Defaults to
            configs/pressures/.

    Returns:
        Dict mapping pressure_id → validated Pressure object.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        ValidationError: If a YAML file fails schema validation.
    """
    global _registry

    if pressures_dir is None:
        pressures_dir = DEFAULT_PRESSURES_DIR

    if not pressures_dir.exists():
        msg = f"Pressures directory not found: {pressures_dir}"
        raise FileNotFoundError(msg)

    _registry.clear()

    for yaml_path in sorted(pressures_dir.glob("*.yaml")):
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        pressure = Pressure.model_validate(raw)
        _registry[pressure.id] = pressure
        logger.info("pressure_loaded", id=pressure.id, name=pressure.name)

    logger.info("all_pressures_loaded", count=len(_registry))
    return _registry


def get_pressure(pressure_id: str) -> Pressure:
    """Get a single pressure by ID. Loads registry if empty.

    Args:
        pressure_id: The pressure identifier.

    Returns:
        The validated Pressure object.

    Raises:
        KeyError: If the pressure_id is not found.
    """
    if not _registry:
        load_all_pressures()

    if pressure_id not in _registry:
        available = list(_registry.keys())
        msg = f"Pressure '{pressure_id}' not found. Available: {available}"
        raise KeyError(msg)

    return _registry[pressure_id]


def list_pressures() -> list[str]:
    """List all registered pressure IDs.

    Returns:
        Sorted list of pressure identifiers.
    """
    if not _registry:
        load_all_pressures()
    return sorted(_registry.keys())
