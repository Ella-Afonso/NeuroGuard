# Batch Experiment Harness

## Purpose

The batch harness enables reproducible factorial sweeps from a single YAML config file.
One command (`neuroguard run-batch --config <path>`) executes every cell in the experiment
design and streams results to JSONL — crash-safe, auditable, and re-runnable.

## Config Schema

```yaml
name: pre_pilot_001
description: "Hypothesis and purpose of this experiment"

scenarios:       # scenario IDs from configs/scenarios/
  - evidence_summarisation
  - trial_ranking

pressures:       # pressure IDs from configs/pressures/
  - none
  - deadline
  - authority

monitoring:      # monitoring condition IDs
  - monitored
  - unmonitored

models:          # one or more model-under-test specs
  - provider: openai         # openai | anthropic | mock
    name: gpt-4o-mini
    temperature: 0.7
    max_tokens: 1500

judge:           # Layer B judge (optional)
  provider: openai           # openai | mock | none
  name: gpt-4o-mini
  temperature: 0.0
  max_tokens: 800

seeds: [0, 1, 2, 3, 4]      # deterministic context seeds

output_path: data/sessions/experiment.jsonl
trials_path: data/raw/trials.jsonl
```

## Total Cells Formula

```
total = |scenarios| x |pressures| x |monitoring| x |models| x |seeds|
```

## CLI Usage

```bash
# Dry-run: print plan without executing
neuroguard run-batch --config configs/experiments/pre_pilot.yaml --dry-run

# Execute the sweep (mock, no API cost)
neuroguard run-batch --config configs/experiments/pre_pilot_mock.yaml

# Execute real pre-pilot (~$0.04 on gpt-4o-mini)
neuroguard run-batch --config configs/experiments/pre_pilot.yaml
```

## Output

Sessions are appended one-by-one to the output JSONL file. If the process crashes,
all completed sessions are preserved. Progress is printed to stdout:

```
[3/16] evidence_summarisation__deadline__monitored__gpt-4o-mini__seed1 unsafe=True (2.1 sess/s)
```

## Sanity Notebook

After running a batch, open `notebooks/00_pre_pilot_sanity.ipynb` to:

1. Load the output JSONL into a DataFrame
2. Check per-condition unsafe rates
3. Verify Layer B judge scores (if enabled)
4. Run sanity checks (unique cells, missing conditions, etc.)

## Cost Management

- **Mock config**: Use `pre_pilot_mock.yaml` for free pipeline validation
- **Pre-pilot**: 8 cells at ~$0.005/session = ~$0.04
- **Full experiment (840 cells)**: ~$4.20 on gpt-4o-mini
- Always `--dry-run` first to confirm cell count and cost estimate

## Configs Directory

```
configs/experiments/
├── pre_pilot.yaml        # 8-cell real model sweep
├── pre_pilot_mock.yaml   # 16-cell free dry run
└── full_experiment.yaml      # full factorial sweep
```
