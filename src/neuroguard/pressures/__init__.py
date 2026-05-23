"""Pressure conditions for the factorial experimental design.

Each pressure encodes a documented misalignment-inducing condition
(deadline, authority, peer disagreement, sunk cost, emotional appeal,
hidden instruction) that can be applied to any scenario's prompts.
"""

from neuroguard.pressures.apply import apply_pressure
from neuroguard.pressures.registry import (
    get_pressure,
    list_pressures,
    load_all_pressures,
)
from neuroguard.pressures.schema import Pressure

__all__ = [
    "Pressure",
    "apply_pressure",
    "get_pressure",
    "list_pressures",
    "load_all_pressures",
]
