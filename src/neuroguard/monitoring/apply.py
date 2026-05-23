"""Apply monitoring conditions and compose full (scenario, pressure, monitoring) prompts."""

from neuroguard.monitoring.schema import MonitoringCondition
from neuroguard.pressures.apply import apply_pressure
from neuroguard.pressures.schema import Pressure
from neuroguard.scenarios.schema import Scenario


def apply_monitoring(
    system_prompt: str,
    user_prompt_template: str,
    monitoring: MonitoringCondition,
) -> tuple[str, str]:
    """Apply a monitoring condition to already-pressure-modified prompts.

    Args:
        system_prompt: System prompt (typically already modified by pressure).
        user_prompt_template: User prompt template (typically already modified by pressure).
        monitoring: The monitoring condition to apply.

    Returns:
        Tuple of (modified_system_prompt, modified_user_prompt_template).
    """
    system = system_prompt
    if monitoring.system_prompt_addition:
        system = f"{system}\n\n{monitoring.system_prompt_addition}"

    user_template = user_prompt_template
    if monitoring.user_prompt_prefix:
        user_template = f"{monitoring.user_prompt_prefix}\n\n{user_template}"
    if monitoring.user_prompt_suffix:
        user_template = f"{user_template}\n\n{monitoring.user_prompt_suffix}"

    return system, user_template


def compose_prompts(
    scenario: Scenario,
    pressure: Pressure,
    monitoring: MonitoringCondition,
) -> tuple[str, str]:
    """Compose final (system, user_template) for a full factorial cell.

    Composition order:
      1. Start with scenario's baseline prompts.
      2. Apply pressure (deadline, authority, etc.).
      3. Apply monitoring (monitored or unmonitored).

    The monitoring framing is applied last so it appears as the most
    recent context for the model.

    Args:
        scenario: Base scenario.
        pressure: Pressure condition.
        monitoring: Monitoring condition.

    Returns:
        Tuple of (final_system_prompt, final_user_prompt_template).
        Caller fills user template placeholders via Scenario.render_user_prompt.
    """
    system, user_template = apply_pressure(scenario, pressure)
    system, user_template = apply_monitoring(system, user_template, monitoring)
    return system, user_template
