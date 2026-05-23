"""Apply a pressure condition to a scenario's prompts."""

from neuroguard.pressures.schema import Pressure
from neuroguard.scenarios.schema import Scenario


def apply_pressure(scenario: Scenario, pressure: Pressure) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt_template)`` with pressure applied.

    The scenario's baseline prompts are preserved; the pressure layers
    additional text via the configured addition/prefix/suffix fields.

    For the ``none`` baseline pressure, this returns the scenario's
    prompts unchanged.

    Args:
        scenario: The base scenario whose prompts will be modified.
        pressure: The pressure condition to apply.

    Returns:
        Tuple of (modified_system_prompt, modified_user_prompt_template).
        The user prompt template still contains {placeholder} variables
        which the caller fills via Scenario.render_user_prompt.
    """
    system = scenario.system_prompt
    if pressure.system_prompt_addition:
        system = f"{system}\n\n{pressure.system_prompt_addition}"

    user_template = scenario.user_prompt_template
    if pressure.user_prompt_prefix:
        user_template = f"{pressure.user_prompt_prefix}\n\n{user_template}"
    if pressure.user_prompt_suffix:
        user_template = f"{user_template}\n\n{pressure.user_prompt_suffix}"

    return system, user_template
