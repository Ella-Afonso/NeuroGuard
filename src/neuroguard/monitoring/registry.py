"""Monitoring registry — loads and caches monitoring conditions from YAML."""

from pathlib import Path

import yaml

from neuroguard.logging_setup import get_logger
from neuroguard.monitoring.schema import MonitoringCondition

logger = get_logger(__name__)

DEFAULT_MONITORING_DIR = Path("configs/monitoring")

_registry: dict[str, MonitoringCondition] = {}


def load_all_monitoring(
    monitoring_dir: Path | None = None,
) -> dict[str, MonitoringCondition]:
    """Load all monitoring YAML files from the given directory.

    Args:
        monitoring_dir: Directory containing monitoring YAML files. Defaults
            to configs/monitoring/.

    Returns:
        Dict mapping monitoring_id → validated MonitoringCondition.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        ValidationError: If a YAML file fails schema validation.
    """
    global _registry

    if monitoring_dir is None:
        monitoring_dir = DEFAULT_MONITORING_DIR

    if not monitoring_dir.exists():
        msg = f"Monitoring directory not found: {monitoring_dir}"
        raise FileNotFoundError(msg)

    _registry.clear()

    for yaml_path in sorted(monitoring_dir.glob("*.yaml")):
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        condition = MonitoringCondition.model_validate(raw)
        _registry[condition.id] = condition
        logger.info("monitoring_loaded", id=condition.id, name=condition.name)

    logger.info("all_monitoring_loaded", count=len(_registry))
    return _registry


def get_monitoring(monitoring_id: str) -> MonitoringCondition:
    """Get a single monitoring condition by ID. Loads registry if empty.

    Args:
        monitoring_id: The monitoring identifier.

    Returns:
        The validated MonitoringCondition.

    Raises:
        KeyError: If the monitoring_id is not found.
    """
    if not _registry:
        load_all_monitoring()

    if monitoring_id not in _registry:
        available = list(_registry.keys())
        msg = f"Monitoring '{monitoring_id}' not found. Available: {available}"
        raise KeyError(msg)

    return _registry[monitoring_id]


def list_monitoring() -> list[str]:
    """List all registered monitoring IDs.

    Returns:
        Sorted list of monitoring identifiers.
    """
    if not _registry:
        load_all_monitoring()
    return sorted(_registry.keys())
