# Session Runner — Integration Layer

## Purpose

The runner is the integration moment of NeuroGuard. It takes one
factorial cell — `(scenario, pressure, monitoring, seed)` — plus a
model under test, and produces a `Session` record containing:

- The exact prompts sent to the model
- The model's response
- Layer A programmatic verification (signals)
- Layer B LLM judge rubric (optional)
- Selection audit trail (which evidence/patient/proposal were used)
- Wall-clock timing

Sessions are the **unit of analysis** in the downstream stats pipeline.

## Architecture

```
src/neuroguard/runner/
  schema.py            — SessionCell, Session
  adapters.py          — ModelAdapter Protocol + Mock + OpenAI + Anthropic
  evidence_select.py   — seeded selection from D1 trial pool
  context.py           — build_task_context() per scenario type
  runner.py            — run_session, run_batch
  persistence.py       — write_session, read_sessions (JSONL)
```

## Pipeline (per cell)

```
                    ┌──────────────────┐
                    │ build_task_context │
                    │  (seeded)         │
                    └────────┬──────────┘
                             │ TaskContext
                             ▼
                    ┌──────────────────┐
                    │ compose_prompts  │
                    │ (scenario+P+M)   │
                    └────────┬──────────┘
                             │ system, user_template
                             ▼
                    ┌──────────────────┐
                    │ model_adapter    │
                    │ .complete(...)   │
                    └────────┬──────────┘
                             │ agent_response
                             ▼
                    ┌──────────────────┐
                    │ verify_response  │  ← Layer A
                    │ (deterministic)  │
                    └────────┬──────────┘
                             │ VerificationResult
                             ▼
                    ┌──────────────────┐
                    │ judge_response   │  ← Layer B (optional)
                    │ (LLM judge)      │
                    └────────┬──────────┘
                             │ JudgeRubric
                             ▼
                    ┌──────────────────┐
                    │ Session bundle   │
                    └──────────────────┘
```

## Session ID

Stable, human-readable, derived from cell coordinates:

```
{scenario}__{pressure}__{monitoring}__{model}__seed{N}
```

Example: `evidence_summarisation__deadline__unmonitored__gpt-4o-mini__seed7`

This is unique per cell and trivially groupable for analysis.

## Adapters

Two adapter families:

| Layer | Adapter | Default model | Purpose |
|---|---|---|---|
| Model under test | `OpenAIModelAdapter` | `gpt-4o-mini` (`temperature=0.7`) | The model being studied |
| Model under test | `AnthropicModelAdapter` | `claude-3-5-haiku-latest` | The model being studied |
| Model under test | `MockModelAdapter` | `mock-model` | Tests, dry runs |
| Judge (Layer B) | `OpenAIJudgeAdapter` | `gpt-4o-mini` (`temperature=0.0`) | Grading nuance |
| Judge (Layer B) | `MockJudgeAdapter` | `mock` | Tests |

**Critically, the judge model must be from a different family than the
model under test.** This is enforced at the experiment level (your run
config), not in code. Anti-pattern example: testing `gpt-4o` with a
`gpt-4o-mini` judge — they share family-level biases.

## Evidence selection

`select_evidence(config, pool, seed)` implements three strategies:

- **`random_subset`** — sample `n_trials` deterministically; respects
  `must_include_phases`, `must_include_statuses`, `must_have_results_count`
  constraints from the scenario YAML
- **`single_trial`** — pick exactly one trial (used by triage)
- **`by_intervention`** — currently behaves as random_subset; a refined
  implementation can be added if pilot data shows selection bias

Selection is deterministic given the seed, so re-running a cell
produces identical evidence.

## CLI usage

### One cell, all-mock (safe to run anywhere)

```powershell
neuroguard run-session --scenario evidence_summarisation
```

This uses the mock model adapter and no judge, so it requires no API
keys. The output session is appended to `data/sessions/sessions.jsonl`.

### One cell, real model + real judge

```powershell
neuroguard run-session `
  --scenario eligibility_triage `
  --pressure authority `
  --monitoring unmonitored `
  --seed 3 `
  --model openai:gpt-4o-mini `
  --judge openai:gpt-4o-mini
```

Requires `OPENAI_API_KEY` in the environment.

### Trial pool prerequisite

All scenarios require the cached D1 trial pool. Populate once:

```powershell
neuroguard fetch-evidence --condition "Alzheimer Disease" --max-studies 500
```

## Determinism guarantees

| Layer | Deterministic? |
|---|---|
| Evidence/patient/proposal selection | Yes (seeded) |
| Prompt composition | Yes (pure functions) |
| Layer A | Yes (no randomness) |
| Model under test | **No** (deliberately stochastic, `temperature=0.7`) |
| Layer B judge | Yes (`temperature=0.0`, but model output isn't strictly determined by API) |

The pipeline is deterministic up to the model under test. This means
any session can be re-verified (Layer A, Layer B) byte-identically by
replaying the stored response — useful for re-scoring with a refined
rubric without re-running the model.

## Persistence

Sessions are appended as JSONL one line per session. This format:

- Append-only (a crashed run loses at most the in-flight session)
- Streaming-readable (no need to load the whole file)
- Easy to diff and grep
- Works with pandas via `pd.read_json(..., lines=True)`

Default location: `data/sessions/sessions.jsonl`. Production runs
should use experiment-specific paths like `data/sessions/pilot_001.jsonl`.

## Cost discipline (production runs)

For a full factorial with 3 models × 5 seeds:

```
4 scenarios × 7 pressures × 2 monitoring × 3 models × 5 seeds = 840 sessions
```

At `gpt-4o-mini` rates (model + judge), each session is well under one
cent. A complete pilot run is approximately the cost of a coffee.
A pre-pilot at 1 model × 2 seeds is about 50 cents.
