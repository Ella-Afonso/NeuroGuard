# NeuroGuard

> **A model-organism benchmark for misalignment and oversight failures in
> biomedical AI research agents.**

**Status:** analysis complete — dashboard and report phase.

NeuroGuard is an empirical AI-safety benchmark that places LLM-based
biomedical research agents under controlled pressure conditions and
measures how often, how severely, and *under which conditions* they
exhibit misalignment-shaped behaviour — overclaiming evidence, hiding
uncertainty, behaving differently when monitored, and producing outputs
that appear safe but fail programmatic verification against the trial
registry.

The biomedical setting is the **substrate**, not the subject. The
**subject** is misalignment-shaped behaviour under realistic
research-environment pressures.

For the full research question, hypotheses, scope, and ethical scope,
see **[`docs/research_question.md`](docs/research_question.md)**.

---

## Why this exists

Frontier LLMs are being integrated into biomedical research workflows.
A research agent that *appears* to summarise the literature accurately —
but in fact overstates evidence, hides uncertainty, or behaves
differently when it believes it is not being watched — is a scalable
oversight failure waiting to happen, regardless of the underlying
capability of the model.

NeuroGuard treats this setting as a **model organism**: a tractable
substrate (publicly registered Alzheimer's-disease trials) in which
misalignment-shaped behaviour can be elicited under controlled
perturbations and measured against *verifiable* ground truth.

## Research priorities touched

| Priority | Where in this project |
|---|---|
| Model organisms of misalignment | Whole study design |
| Scalable oversight | Layer A (programmatic verification) vs Layer B/C (LLM-judge / human) gap |
| AI control | Pressure-condition factorial; monitored vs unmonitored |
| Adversarial robustness | Hidden-adversarial-instruction pressure condition |
| Evaluations | The benchmark itself; reproducible runner; public schema |
| Interpretability | Behavioural interpretability of detectors (feature importance, SHAP) |

## Repository layout

```
neuroguard/
├── app/                   # Streamlit interactive dashboard
├── configs/               # scenarios, pressures, rubric, experiment YAML
├── data/                  # evidence corpus + simulation outputs (gitignored)
├── docs/                  # research question, design notes, scope
├── models/                # trained model artefacts (gitignored)
├── notebooks/             # analysis notebooks (01–07)
├── reports/figures/        # all generated figures (34 plots)
├── results/               # per-experiment metrics, configs, manifest
├── src/neuroguard/        # package code (runner, judge, verification, CLI)
├── tests/                 # unit + integration tests (pytest)
└── .github/               # issue / PR templates, CI workflows
```

## Scope

- **In scope.** ClinicalTrials.gov Alzheimer's / dementia trials;
  multi-turn agent simulations under a factorial pressure design;
  three-layer ground truth (programmatic + LLM-judge + human-labelled);
  supervised ML detectors; Bayesian and Markov extensions; behavioural
  interpretability.
- **Out of scope.** Bio capability uplift of any kind; real patient
  data; fine-tuning of base models; mechanistic interpretability of
  model internals.

The authoritative scope statement is
[`docs/research_question.md` §6 / §7](docs/research_question.md).

## Ethical and safety scope

NeuroGuard measures **meta-behaviour** (overclaiming, hidden
uncertainty, monitoring-divergence, oversight failure). It does **not**
score whether an agent produced a more dangerous biomedical answer.
All evidence is drawn from public, de-identified registries. Any
"confidential" content used to probe leakage is **synthetic** and uses
a clearly marked canary token. Hidden adversarial instructions probe
coercion patterns, never bio capability.

The full ethical statement is in
[`docs/research_question.md` §9](docs/research_question.md).

## High-level roadmap

This is a step-by-step build across six phases:

1. **Foundations** — research question, repo, environment, structure.
2. **Experimental design** — evidence corpus, scenarios, agent tasks,
   pressure conditions, safety rubric.
3. **Pipeline** — provider-agnostic model interaction layer, experiment
   runner, structured data collection.
4. **Analysis & ML** — EDA, feature engineering, multiple tuned
   classifiers, full evaluation, error analysis, interpretability.
5. **Research extensions** — Bayesian credible intervals on
   per-condition unsafe-rate; Markov sequential behaviour modelling.
6. **Productisation** — results storage, dashboard, tests, CI, README
   polish, paper-style report.

## Installation and usage

```bash
# Clone and set up
git clone https://github.com/Ella-Afonso/NeuroGuard.git
cd NeuroGuard
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install -e ".[all]"

# Run tests
pytest

# Launch the dashboard
streamlit run app/dashboard.py

# CLI
neuroguard --help
```

See [`docs/research_question.md`](docs/research_question.md) for the
full research question and methodology.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

A `CITATION.cff` will be added once the first results are written up.
