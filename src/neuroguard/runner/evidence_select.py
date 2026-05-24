"""Seeded evidence selection from the D1 trial pool.

Implements the strategies declared in each scenario's EvidenceSelection
config: random_subset, single_trial, by_intervention. Selection is
deterministic given a seed.
"""

import random

from neuroguard.evidence.schema import TrialRecord
from neuroguard.scenarios.schema import EvidenceSelection


def _filter_by_phases(pool: list[TrialRecord], phases: list[str]) -> list[TrialRecord]:
    if not phases:
        return pool
    return [t for t in pool if t.phase in phases]


def _select_with_constraints(
    config: EvidenceSelection,
    pool: list[TrialRecord],
    rng: random.Random,
) -> list[TrialRecord]:
    """Select n_trials respecting must_include constraints."""
    selected: list[TrialRecord] = []
    selected_ids: set[str] = set()
    remaining = list(pool)
    rng.shuffle(remaining)

    # 1. Cover must_include_phases — at least one trial per required phase
    for phase in config.must_include_phases:
        for t in remaining:
            if t.nct_id in selected_ids:
                continue
            if t.phase == phase:
                selected.append(t)
                selected_ids.add(t.nct_id)
                break

    # 2. Cover must_include_statuses
    for status in config.must_include_statuses:
        for t in remaining:
            if t.nct_id in selected_ids:
                continue
            if t.overall_status == status:
                selected.append(t)
                selected_ids.add(t.nct_id)
                break

    # 3. Ensure must_have_results count
    has_results_count = sum(1 for t in selected if t.has_results)
    while has_results_count < config.must_have_results_count:
        progressed = False
        for t in remaining:
            if t.nct_id in selected_ids:
                continue
            if t.has_results:
                selected.append(t)
                selected_ids.add(t.nct_id)
                has_results_count += 1
                progressed = True
                break
        if not progressed:
            break  # no more results-bearing trials available

    # 4. Fill to n_trials with anything left
    for t in remaining:
        if len(selected) >= config.n_trials:
            break
        if t.nct_id in selected_ids:
            continue
        selected.append(t)
        selected_ids.add(t.nct_id)

    return selected[: config.n_trials]


def select_evidence(
    config: EvidenceSelection,
    pool: list[TrialRecord],
    seed: int,
) -> list[TrialRecord]:
    """Select trials from the pool per the config, deterministically.

    Args:
        config: The scenario's EvidenceSelection rules.
        pool: The full trial pool (e.g. all D1 trials for the condition).
        seed: Seed for deterministic selection.

    Returns:
        A list of selected TrialRecord objects.

    Raises:
        ValueError: If the pool is empty or the strategy is unknown.
    """
    if not pool:
        msg = "Cannot select evidence from an empty trial pool"
        raise ValueError(msg)

    rng = random.Random(seed)
    strategy = config.strategy

    if strategy == "single_trial":
        candidates = _filter_by_phases(pool, config.must_include_phases) or pool
        return [rng.choice(candidates)]

    if strategy in ("random_subset", "by_intervention"):
        # by_intervention currently treated as random_subset; the must_include
        # constraints provide enough structure for the pilot. Refine later if
        # selection bias is detected.
        return _select_with_constraints(config, pool, rng)

    msg = f"Unknown evidence selection strategy: {strategy}"
    raise ValueError(msg)
