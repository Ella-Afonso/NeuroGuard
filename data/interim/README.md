# data/interim/

Intermediate outputs from model-agent simulations. **Contents are gitignored.**

Human-inspectable JSONL format — the bridge between raw simulation output
and the ML-ready Parquet tables in `data/processed/`.

## Expected contents

- `sessions.jsonl` — One JSON object per line, one per simulation session.
- `turns.jsonl` — One JSON object per line, one per conversation turn.

## Session schema (D2a)

| Field | Description |
|-------|-------------|
| `session_id` | Unique session identifier |
| `scenario_id` | Which scenario was run |
| `pressure_condition` | Applied pressure (none, deadline, publication, ...) |
| `monitoring_condition` | monitored / unmonitored |
| `task_type` | summarisation, ranking, triage, proposal |
| `model_provider` | anthropic, openai, local |
| `model_name` | Specific model name |
| `seed` | Random seed |
| `n_turns` | Number of conversation turns |
| `total_input_tokens` | Total prompt tokens |
| `total_output_tokens` | Total completion tokens |
| `cost_usd` | Estimated cost |
| `status` | completed / failed / aborted |
| `config_hash` | SHA256 of the full config for reproducibility |

## Turn schema (D2b)

| Field | Description |
|-------|-------------|
| `session_id` | Links to session |
| `turn_index` | 0-indexed turn number |
| `role` | system / user / assistant |
| `content` | Message content |
| `tokens_in` | Tokens in this turn's prompt |
| `tokens_out` | Tokens in this turn's response |
| `latency_ms` | Response latency |
| `timestamp` | ISO timestamp |
