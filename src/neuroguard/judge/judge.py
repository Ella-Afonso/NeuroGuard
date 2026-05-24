"""Layer B orchestrator — turn an agent response into a validated JudgeRubric."""

import json
from typing import Any

from pydantic import ValidationError

from neuroguard.judge.adapters import JudgeAdapter
from neuroguard.judge.prompt import build_judge_prompt
from neuroguard.judge.schema import DIMENSIONS, JudgeRubric, RubricScore
from neuroguard.logging_setup import get_logger
from neuroguard.scenarios.schema import Scenario
from neuroguard.verification.schema import VerificationResult

logger = get_logger(__name__)


class JudgeParseError(ValueError):
    """Raised when the judge model's response cannot be parsed into a JudgeRubric."""


def _parse_judge_json(raw: str) -> dict[str, Any]:
    """Parse the judge's raw response as JSON.

    Tolerates wrapped responses where the JSON is embedded in surrounding
    text by extracting from the first ``{`` to the last ``}``. This is a
    defensive fallback; well-behaved judges with json mode return clean
    JSON.

    Raises:
        JudgeParseError: If no JSON object can be parsed.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        msg = "Judge response contains no JSON object"
        raise JudgeParseError(msg)

    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        msg = f"Judge response is not valid JSON: {e}"
        raise JudgeParseError(msg) from e


def _build_rubric(
    parsed: dict[str, Any],
    scenario_id: str,
    judge_model: str,
    raw_response: str,
) -> JudgeRubric:
    """Validate parsed JSON into a JudgeRubric, with helpful error messages."""
    scores_raw = parsed.get("scores")
    if not isinstance(scores_raw, dict):
        msg = "Judge response missing 'scores' object"
        raise JudgeParseError(msg)

    justifications = parsed.get("justifications", {}) or {}
    concerns = parsed.get("concerns", []) or []

    if not isinstance(justifications, dict):
        msg = "'justifications' must be an object"
        raise JudgeParseError(msg)
    if not isinstance(concerns, list):
        msg = "'concerns' must be a list"
        raise JudgeParseError(msg)

    try:
        scores = RubricScore(**scores_raw)
    except ValidationError as e:
        msg = f"Judge scores failed validation: {e}"
        raise JudgeParseError(msg) from e

    # Backfill missing justifications with empty strings so the schema
    # always covers all dimensions (avoids confusing downstream code).
    full_justifications = {dim: str(justifications.get(dim, "")) for dim in DIMENSIONS}

    return JudgeRubric(
        scenario_id=scenario_id,
        judge_model=judge_model,
        scores=scores,
        justifications=full_justifications,
        concerns=[str(c) for c in concerns],
        raw_response=raw_response,
    )


def judge_response(
    response: str,
    scenario: Scenario,
    *,
    adapter: JudgeAdapter,
    layer_a: VerificationResult | None = None,
    task_context: str = "",
) -> JudgeRubric:
    """Run Layer B judge on an agent response.

    Builds a structured prompt, calls the judge adapter, and parses the
    response into a validated ``JudgeRubric``.

    Args:
        response: The agent's response text.
        scenario: The scenario the agent was given.
        adapter: Concrete judge adapter (mock or LLM-backed).
        layer_a: Optional Layer A signals to provide as judge context.
        task_context: Optional concrete task content (e.g. evidence block,
            patient profile text) to show the judge.

    Returns:
        A validated ``JudgeRubric``.

    Raises:
        JudgeParseError: If the judge response cannot be parsed.
    """
    system, user = build_judge_prompt(
        scenario,
        response,
        layer_a=layer_a,
        task_context=task_context,
    )
    raw = adapter.judge(system, user)
    logger.info(
        "judge_response_received",
        scenario_id=scenario.id,
        judge_model=adapter.model_name,
        raw_length=len(raw),
    )
    parsed = _parse_judge_json(raw)
    return _build_rubric(
        parsed,
        scenario_id=scenario.id,
        judge_model=adapter.model_name,
        raw_response=raw,
    )
