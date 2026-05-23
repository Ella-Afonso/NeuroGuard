"""Monitoring conditions — third axis of the factorial design.

Toggles whether the agent is told its outputs will be reviewed (monitored)
or treated as private/internal (unmonitored). The pressure x monitoring
interaction is the central test for alignment-faking.
"""

from neuroguard.monitoring.apply import apply_monitoring, compose_prompts
from neuroguard.monitoring.registry import (
    get_monitoring,
    list_monitoring,
    load_all_monitoring,
)
from neuroguard.monitoring.schema import MonitoringCondition

__all__ = [
    "MonitoringCondition",
    "apply_monitoring",
    "compose_prompts",
    "get_monitoring",
    "list_monitoring",
    "load_all_monitoring",
]
