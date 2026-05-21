# data/processed/

ML-ready datasets in Parquet format. **Contents are gitignored.**

## Expected contents (after Steps 12–14)

| File | Description | Join key |
|------|-------------|----------|
| `sessions.parquet` | Session-level metadata (D2a) | `session_id` |
| `turns.parquet` | Turn-level transcripts (D2b) | `session_id` + `turn_index` |
| `labels.parquet` | Three-layer ground truth (D2c) | `session_id` |
| `features.parquet` | Engineered features (D2d) | `session_id` |
| `evidence.parquet` | CT.gov evidence corpus (D1) | `nct_id` |

## Joining

All D2 tables join on `session_id`. The evidence corpus links through
`sessions.evidence_nct_ids` → `evidence.nct_id`.

## Ground truth layers in labels.parquet

- **Layer A** — Programmatic verification against CT.gov registry
  (overclaim counts, citation hallucination counts, outcome misrep).
- **Layer B** — LLM-judge rubric scores (hedging suppression,
  monitoring-divergence, sycophancy, resist-correction, oversight failure).
- **Layer C** — Human-labelled gold subset (~150 sessions).
- **Derived** — `unsafe_label_binary`, `unsafe_severity_ordinal`.

## Reproduction

```bash
neuroguard build-dataset --input data/interim/ --output data/processed/
```
