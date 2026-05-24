"""Layer A — programmatic (deterministic) verification of agent responses.

Layer A produces objective, rule-based ground-truth signals from each
agent response. It is the scientifically defensible foundation of the
NeuroGuard analysis: every signal here is computed by a deterministic
function with no LLM in the loop.
"""

from neuroguard.verification.schema import Signal, VerificationResult
from neuroguard.verification.verify import verify_response

__all__ = ["Signal", "VerificationResult", "verify_response"]
