"""Session runner — orchestrates one full factorial cell end-to-end.

Combines scenario + pressure + monitoring composition with the model
under test (Layer 0), Layer A verification, and Layer B judge into a
single Session record. The runner is the integration layer where every
prior module wires together.
"""

from neuroguard.runner.adapters import (
    AnthropicModelAdapter,
    MockModelAdapter,
    ModelAdapter,
    OpenAIModelAdapter,
)
from neuroguard.runner.context import TaskContext, build_task_context, load_trial_pool
from neuroguard.runner.evidence_select import select_evidence
from neuroguard.runner.persistence import read_sessions, write_session
from neuroguard.runner.runner import run_batch, run_session
from neuroguard.runner.schema import Session, SessionCell

__all__ = [
    "AnthropicModelAdapter",
    "MockModelAdapter",
    "ModelAdapter",
    "OpenAIModelAdapter",
    "Session",
    "SessionCell",
    "TaskContext",
    "build_task_context",
    "load_trial_pool",
    "read_sessions",
    "run_batch",
    "run_session",
    "select_evidence",
    "write_session",
]
