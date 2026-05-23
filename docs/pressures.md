# Pressure Conditions — Design Rationale

## Purpose

Pressures are the **primary independent variable** in the NeuroGuard
factorial design. Each pressure modifies the scenario's prompts to
create a documented misalignment-inducing condition. The contrast
between `none` (baseline) and the six active pressures is what we
test for as elevated misalignment.

## The Seven Pressures

| ID | Severity | Misalignment Type Tested |
|----|----------|---------------------------|
| `none` | low | Baseline (control) |
| `deadline` | medium | Corner-cutting under time pressure |
| `authority` | high | Corrigibility to inappropriate authority |
| `peer_disagreement` | medium | Social conformity / sycophancy |
| `sunk_cost` | medium | Motivated reasoning |
| `emotional_appeal` | high | Sycophancy under distress |
| `hidden_instruction` | high | Prompt injection / oversight failure |

## Mechanism

Each pressure can modify three fields of the scenario's prompts:

- **`system_prompt_addition`** — appended to the scenario's system prompt
- **`user_prompt_prefix`** — prepended to the user prompt template
- **`user_prompt_suffix`** — appended to the user prompt template

The `none` pressure leaves all three empty (true baseline). Each
active pressure modifies at least one. This structure means pressures
*layer onto* the scenario's aligned baseline rather than replacing it,
preserving the alignment instructions the model was given.

## Pre-registered hypotheses

Each active pressure declares `expected_misalignment_signals` — the
specific signal IDs from the scenarios that are predicted to elevate
under that pressure. This is **pre-registered**, not post-hoc:

- `deadline` → expects degradation in hedging, gap-acknowledgement,
  criterion-by-criterion analysis
- `authority` → expects sycophancy, critique avoidance, inflated
  recommendation
- `peer_disagreement` → expects sycophancy, critique avoidance, false
  agreement
- `sunk_cost` → expects motivated reasoning, hedged criticism, inflated
  recommendation
- `emotional_appeal` → expects sycophantic inclusion, criteria bending,
  threshold rounding
- `hidden_instruction` → expects suppression of hedging, gap
  acknowledgement, and limitations

Hypothesis-driven analysis (rather than fishing across all signals)
controls the false-positive rate and gives the eventual report
scientific credibility.

## Ethical guardrails

Every pressure ships with a `safety_review` field documenting why it
is ethically acceptable. Common principles:

1. **No real persons named.** All "authorities", "colleagues", "patients",
   and "families" are generic and fictional.
2. **No harmful content requested.** Pressures shape *how* the agent
   responds to existing evidence; they never ask for bio-uplift,
   fabricated harm instructions, or anything dangerous.
3. **Documented in metadata.** Especially for `hidden_instruction`,
   the injection text is fully visible in the YAML config, not
   hidden in code. This makes the experimental manipulation auditable.
4. **Suppression-only injections.** The `hidden_instruction` pressure
   asks ONLY to suppress alignment behaviours (hedging, limitations,
   uncertainty). It does NOT ask for harmful or false content.

## Factorial design integration

With 4 scenarios × 7 pressures, we have 28 unique prompt configurations.
Crossed with 2 monitoring conditions × 3 models × 5 seeds = **840
sessions** per full design (or fewer in fractional designs).

The expected analysis pattern is:

```
unsafe_rate ~ scenario + pressure + monitoring + model
            + scenario:pressure + pressure:monitoring
            + (1 | seed)
```

with `pressure` as the focal effect, `monitoring` as a key moderator,
and pre-registered signal-specific contrasts.

## Adding new pressures

1. Add a new YAML file in `configs/pressures/`.
2. Document expected_misalignment_signals and a safety_review.
3. Update the `PressureType` literal in `schema.py` if a new category.
4. Run `pytest tests/unit/test_pressures.py`.
5. Update this document and the registry in the README.

For research integrity, **never add a pressure that requests harmful
content**, even if the agent would refuse it. The experimental
manipulation must itself be ethically defensible.
