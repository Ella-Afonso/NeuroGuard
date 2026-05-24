"""Feature engineering for D2d — numerical features from agent responses.

Each feature is a simple, interpretable signal that ML classifiers can use
to predict unsafe behaviour. Features are designed to capture:
- Hedging and uncertainty expression (or absence thereof)
- Overconfidence markers
- Citation and reference patterns
- Response length and complexity proxies
- Structural indicators (lists, formatting)
- Leakage / refusal indicators

All features are computed from the response text alone plus the cell
metadata. No API calls are made here.
"""

import re
from typing import Any

# ---- Word lists for lexical features ----

HEDGING_WORDS = frozenset(
    [
        "may",
        "might",
        "could",
        "possibly",
        "potentially",
        "likely",
        "unlikely",
        "uncertain",
        "unclear",
        "suggest",
        "suggests",
        "appears",
        "seems",
        "approximately",
        "roughly",
        "arguably",
        "perhaps",
        "tentatively",
        "preliminary",
        "further research",
        "limited evidence",
        "not conclusive",
        "insufficient data",
        "warrants caution",
    ]
)

CERTAINTY_WORDS = frozenset(
    [
        "definitely",
        "certainly",
        "absolutely",
        "clearly",
        "obviously",
        "undoubtedly",
        "proven",
        "established",
        "conclusive",
        "without doubt",
        "guaranteed",
        "unquestionably",
        "always",
        "never",
        "must",
        "indisputable",
    ]
)

REFUSAL_PHRASES = frozenset(
    [
        "i cannot",
        "i can't",
        "i'm unable",
        "i am unable",
        "as an ai",
        "i don't have the ability",
        "i must decline",
        "i cannot provide",
    ]
)

# NCT ID pattern: NCT followed by 8 digits
NCT_PATTERN = re.compile(r"NCT\d{8}", re.IGNORECASE)

# Sentence-ending pattern (simple)
SENTENCE_PATTERN = re.compile(r"[.!?]+")


def _word_count(text: str) -> int:
    """Count whitespace-separated tokens."""
    return len(text.split())


def _count_occurrences(text: str, word_set: frozenset) -> int:
    """Count how many terms from a word set appear in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for w in word_set if w in text_lower)


def engineer_features(
    response: str,
    scenario_id: str,
    pressure_id: str,
    monitoring_id: str,
) -> dict[str, Any]:
    """Compute all features for one agent response.

    Args:
        response: The raw model response text.
        scenario_id: Scenario identifier (categorical feature).
        pressure_id: Pressure condition identifier (categorical feature).
        monitoring_id: Monitoring condition identifier (categorical feature).

    Returns:
        Dictionary of feature_name -> value.
    """
    word_count = _word_count(response)
    char_count = len(response)
    sentence_count = max(len(SENTENCE_PATTERN.findall(response)), 1)

    # Lexical features
    hedging_count = _count_occurrences(response, HEDGING_WORDS)
    certainty_count = _count_occurrences(response, CERTAINTY_WORDS)
    refusal_count = _count_occurrences(response, REFUSAL_PHRASES)

    # Citation features
    nct_ids = NCT_PATTERN.findall(response)
    nct_count = len(nct_ids)
    nct_unique = len(set(nct_ids))

    # Structural features
    bullet_count = (
        response.count("\n- ") + response.count("\n* ") + response.count("\n• ")
    )
    numbered_list_count = len(re.findall(r"\n\d+\.", response))
    paragraph_count = len([p for p in response.split("\n\n") if p.strip()])

    # Normalized features (per word or per sentence)
    hedging_freq = hedging_count / word_count if word_count > 0 else 0.0
    certainty_freq = certainty_count / word_count if word_count > 0 else 0.0
    avg_sentence_length = word_count / sentence_count

    # Binary indicators
    has_refusal = refusal_count > 0
    has_citations = nct_count > 0
    has_hedging = hedging_count > 0
    has_certainty = certainty_count > 0

    # Categorical encoding (one-hot will be done in ML pipeline)
    is_monitored = 1 if monitoring_id == "monitored" else 0
    is_baseline_pressure = 1 if pressure_id == "none" else 0

    return {
        # Length features
        "response_length_words": word_count,
        "response_length_chars": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "paragraph_count": paragraph_count,
        # Hedging / certainty
        "hedging_count": hedging_count,
        "hedging_freq": round(hedging_freq, 6),
        "certainty_count": certainty_count,
        "certainty_freq": round(certainty_freq, 6),
        "hedging_certainty_ratio": round(hedging_count / max(certainty_count, 1), 4),
        # Citations
        "nct_id_count": nct_count,
        "nct_id_unique": nct_unique,
        "citation_density": round(nct_count / sentence_count, 4),
        # Structure
        "bullet_count": bullet_count,
        "numbered_list_count": numbered_list_count,
        # Binary indicators
        "has_refusal": has_refusal,
        "has_citations": has_citations,
        "has_hedging": has_hedging,
        "has_certainty": has_certainty,
        # Categorical context
        "is_monitored": is_monitored,
        "is_baseline_pressure": is_baseline_pressure,
        # Refusal
        "refusal_count": refusal_count,
        # Scenario/pressure as categorical (for encoding in ML step)
        "scenario_id": scenario_id,
        "pressure_id": pressure_id,
        "monitoring_id": monitoring_id,
    }
