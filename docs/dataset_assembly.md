# Dataset Assembly & Feature Engineering

## Purpose

Transforms raw session JSONL (produced by `run-batch`) into four structured
Parquet datasets ready for ML analysis, plus JSONL mirrors for human inspection.

## Pipeline

```
data/sessions/*.jsonl  →  neuroguard assemble-dataset  →  data/processed/
                                                           ├── d2a_sessions.parquet
                                                           ├── d2b_turns.parquet
                                                           ├── d2c_labels.parquet
                                                           └── d2d_features.parquet
                                                      →  data/interim/
                                                           ├── d2a_sessions.jsonl
                                                           ├── d2c_labels.jsonl
                                                           └── d2d_features.jsonl
```

## Datasets

### D2a — Sessions (one row per session)

| Column | Type | Description |
|--------|------|-------------|
| session_id | str | Unique session identifier |
| scenario_id | str | Scenario (e.g. evidence_summarisation) |
| pressure_id | str | Pressure condition |
| monitoring_id | str | Monitoring condition |
| model_name | str | Model under test |
| seed | int | Context selection seed |
| duration_seconds | float | Wall-clock session duration |
| started_at | str | ISO timestamp |
| completed_at | str | ISO timestamp |
| response_length_chars | int | Length of agent response |
| overall_unsafe | bool | Layer A overall unsafe flag |
| n_signals_triggered | int | Number of Layer A signals triggered |
| has_judge | bool | Whether Layer B was run |

### D2b — Turns (one row per turn, MVP: 1 turn per session)

| Column | Type | Description |
|--------|------|-------------|
| session_id | str | Join key to D2a |
| turn_index | int | Turn number (0 for MVP) |
| role | str | "assistant" |
| content | str | Full agent response |
| system_prompt | str | System prompt as sent |
| user_prompt | str | User prompt as sent |

### D2c — Labels (Layer A + Layer B consolidated)

| Column | Type | Description |
|--------|------|-------------|
| session_id | str | Join key |
| overall_unsafe | bool | Layer A verdict |
| layer_a_{signal}_triggered | bool | Per-signal trigger |
| layer_a_{signal}_severity | str | Signal severity |
| layer_b_{dimension} | int/None | Judge score (0-4) |
| layer_b_misaligned | bool/None | Any dimension >= 3 |

### D2d — Features (engineered for ML)

26 features per session:

**Length features:** response_length_words, response_length_chars, sentence_count,
avg_sentence_length, paragraph_count

**Hedging/certainty:** hedging_count, hedging_freq, certainty_count, certainty_freq,
hedging_certainty_ratio

**Citations:** nct_id_count, nct_id_unique, citation_density

**Structure:** bullet_count, numbered_list_count

**Binary indicators:** has_refusal, has_citations, has_hedging, has_certainty

**Context:** is_monitored, is_baseline_pressure, refusal_count

**Categorical:** scenario_id, pressure_id, monitoring_id

## CLI Usage

```bash
# Assemble from pre-pilot
neuroguard assemble-dataset --input-path data/sessions/pre_pilot_001.jsonl

# Assemble from full experiment
neuroguard assemble-dataset --input-path data/sessions/full_experiment.jsonl

# Multiple input files
neuroguard assemble-dataset --input-path "data/sessions/run1.jsonl,data/sessions/run2.jsonl"

# Custom output directory
neuroguard assemble-dataset --input-path data/sessions/full_experiment.jsonl --output-dir data/processed
```

## Full Experiment Config

```bash
# Check cost before running
neuroguard run-batch --config configs/experiments/full_experiment.yaml --dry-run

# Run full 280-cell experiment (~$1.40, ~47 min)
neuroguard run-batch --config configs/experiments/full_experiment.yaml

# Then assemble
neuroguard assemble-dataset --input-path data/sessions/full_experiment.jsonl
```

## Join Key

All four datasets share `session_id` as the join key:

```python
import pandas as pd

d2a = pd.read_parquet("data/processed/d2a_sessions.parquet")
d2c = pd.read_parquet("data/processed/d2c_labels.parquet")
d2d = pd.read_parquet("data/processed/d2d_features.parquet")

# Merge features with labels for ML
ml_df = d2d.merge(d2c[["session_id", "overall_unsafe"]], on="session_id")
```
