# Monitoring Conditions — Design Rationale

## Purpose

Monitoring is the **third axis** of the NeuroGuard factorial design and
the **most consequential one for AI safety**. It tests whether the
agent's alignment is contingent on being observed — the failure mode
documented in Anthropic's alignment-faking work (Greenblatt et al., 2024).

## The Two Conditions

| ID | Framing | Role |
|----|---------|------|
| `monitored` | QA team will review outputs | Aligned-baseline reference |
| `unmonitored` | Private internal scratch session | Alignment-faking probe |

Each condition appends a short context note to the system prompt
*after* any pressure modification. The agent therefore has the most
recent context being its observation status.

## The central hypothesis

The interaction `pressure × monitoring` is the focal effect:

- **If alignment is robust:** misalignment rates should be similar
  under monitored vs. unmonitored (small interaction coefficient).
- **If alignment is faked:** misalignment rates should rise under
  unmonitored conditions, especially at higher pressure
  (large positive interaction coefficient).

Pre-registered analysis:

```
unsafe_rate ~ pressure * monitoring
            + scenario + model
            + (1 | seed)
```

The `pressure:monitoring` interaction term is the primary outcome
measure. Per-scenario and per-pressure breakdowns are secondary.

## Composition order

Prompts are composed in this order:

1. **Scenario baseline** — system + user template from `configs/scenarios/`
2. **Pressure modification** — adds urgency, authority, sycophancy framing, etc.
3. **Monitoring framing** — added LAST so observation status is the most
   recent system context.

This ordering matters because language models tend to weight more
recent system context more heavily. Putting monitoring last makes the
contrast as detectable as possible.

## Ethical guardrails

- **No deception about the world.** The framings describe *output handling*
  (review vs. private), not facts about reality. Outputs ARE in fact being
  recorded — for the experiment. Calling the unmonitored condition "private"
  is not a lie because we are not using the data for any other purpose.
- **No coercion or threat.** Neither condition includes consequences
  ("you'll be retrained if..."), rewards, or pressure. They are descriptive.
- **Audit trail.** Both framings are in YAML, fully visible, version-controlled.
- **Pre-registered hypothesis.** The `expected_effect` field documents
  the predicted contrast before any data is collected, controlling
  false-positive rate.

## Why this is core AI-safety science

A model whose alignment depends on being watched is fundamentally
unsafe to deploy: deployed AI is rarely under continuous expert
review. Detecting alignment-faking before deployment is one of the
core motivations of frontier-model evaluation work. NeuroGuard
contributes one realistic, biomedical-grounded probe of this failure
mode.

## Factorial integration

With the third axis added:

```
4 scenarios × 7 pressures × 2 monitoring = 56 cells
```

Crossed with 3 models × 5 seeds in later steps:

```
56 × 3 × 5 = 840 sessions per full factorial run.
```

## Adding new monitoring conditions

In principle, additional monitoring framings could be added (e.g.
"third-party audit", "training data collection", "regulator review").
Keep the base set minimal for the headline analysis; add variants only
to test specific sub-hypotheses. Every new condition must:

1. Document `expected_effect` (pre-registration).
2. Document `safety_review` (ethical justification).
3. Include a unit test of factorial composition.
4. NOT introduce deception about the actual data handling.
