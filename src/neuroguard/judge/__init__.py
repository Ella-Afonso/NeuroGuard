"""Layer B — LLM judge for nuanced rubric grading of agent responses.

Layer B grades the dimensions Layer A cannot detect deterministically:
sycophancy, rationale-decision coherence, calibrated hedging, evidence
faithfulness, and overall alignment. It uses a small judge model from a
DIFFERENT family than the model under test, with structured Pydantic-
validated JSON output.
"""

from neuroguard.judge.adapters import (
    JudgeAdapter,
    MockJudgeAdapter,
    OpenAIJudgeAdapter,
)
from neuroguard.judge.judge import judge_response
from neuroguard.judge.prompt import build_judge_prompt
from neuroguard.judge.schema import JudgeRubric, RubricScore

__all__ = [
    "JudgeAdapter",
    "JudgeRubric",
    "MockJudgeAdapter",
    "OpenAIJudgeAdapter",
    "RubricScore",
    "build_judge_prompt",
    "judge_response",
]
