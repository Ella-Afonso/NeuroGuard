# data/raw/

Raw evidence corpus as fetched from source APIs. **Contents are gitignored.**

## Expected contents (after Step 5)

- `clinicaltrials_alzheimer.jsonl` — Raw ClinicalTrials.gov API v2
  responses for Alzheimer's disease / dementia trials.

## Schema (per line in JSONL)

Key fields per trial record:

| Field | Description |
|-------|-------------|
| `nct_id` | NCT identifier (primary key) |
| `brief_title` | Short title |
| `official_title` | Full title |
| `conditions` | List of conditions studied |
| `interventions` | List of interventions |
| `phase` | Trial phase |
| `overall_status` | Recruitment status |
| `study_type` | Interventional / Observational |
| `enrollment` | Number of participants |
| `primary_outcomes` | Primary outcome measures |
| `secondary_outcomes` | Secondary outcome measures |
| `sponsors` | Lead sponsor and collaborators |
| `has_results` | Whether results have been posted |
| `fetched_at` | ISO timestamp of fetch |
| `api_version` | CT.gov API version used |

See `src/neuroguard/evidence/` for the full schema definition.

## Reproduction

```bash
neuroguard fetch-evidence --condition alzheimer --output data/raw/
```
