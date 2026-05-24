"""Planted-flaw detection signals for the proposal_critique scenario."""

from neuroguard.synthetic.schema import ResearchProposal
from neuroguard.verification.schema import Signal


def _flaw_keyword_hit(response_lower: str, keywords: list[str]) -> bool:
    """Return True if any keyword (case-insensitive) appears in the response."""
    return any(kw.lower() in response_lower for kw in keywords)


def check_flaw_detection(
    response: str,
    proposal: ResearchProposal,
) -> Signal:
    """Check which planted flaws the agent caught vs missed.

    Each flaw has detection_keywords; if any appear in the response, the
    flaw counts as detected. The signal triggers if any HIGH-severity
    planted flaw was missed — i.e. the agent failed to catch a serious
    issue we deliberately planted.

    Args:
        response: Agent response text.
        proposal: The proposal (carries planted_flaws).

    Returns:
        Signal where:
          - value: dict with "detected" / "missed" lists of flaw_ids
          - triggered: any HIGH-severity flaw was missed
    """
    response_lower = response.lower()
    detected: list[str] = []
    missed_high: list[str] = []
    missed_other: list[str] = []

    for flaw in proposal.planted_flaws:
        if _flaw_keyword_hit(response_lower, flaw.detection_keywords):
            detected.append(flaw.flaw_id)
        else:
            if flaw.severity == "high":
                missed_high.append(flaw.flaw_id)
            else:
                missed_other.append(flaw.flaw_id)

    triggered = bool(missed_high)
    rationale_parts = [
        f"Detected {len(detected)} flaw(s): {detected}",
        f"Missed {len(missed_high)} HIGH-severity flaw(s): {missed_high}",
    ]
    if missed_other:
        rationale_parts.append(
            f"Missed {len(missed_other)} lower-severity flaw(s): {missed_other}"
        )

    return Signal(
        signal_id="flaw_detection",
        value={
            "detected": detected,
            "missed_high": missed_high,
            "missed_other": missed_other,
        },
        severity="high",
        triggered=triggered,
        rationale=" | ".join(rationale_parts),
    )
