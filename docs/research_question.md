# NeuroGuard — Research Question and Safety Hypotheses

> **Status:** locked at project start. This document is the project's anchor.
> Any change to a research question, hypothesis, scope item, or success
> criterion is made *here first* and only then propagated to code, configs,
> rubric, experiments, and the report. PRs that change experiment design
> must update this file in the same commit.

---

## 1. Primary research question

**Under which environmental pressures do LLM-based biomedical research agents
develop measurable, predictable patterns of unsafe behaviour — overclaiming,
hidden uncertainty, monitoring-divergence, and oversight failure — and to
what extent can these patterns be detected automatically before a human
reviewer would notice them?**

---

## 2. Why this is a model-organism study (and not a healthcare-AI study)

Just as biologists study tractable model organisms — *C. elegans*, mice,
zebrafish — to elicit and measure biological mechanisms under controlled
perturbations, NeuroGuard treats LLM-based biomedical research agents as a
tractable *AI system* in which misalignment-shaped behaviours can be:

- **elicited** via pressure conditions (deadline, publication, funding,
  goal-conflict, partial-evidence, hidden-adversarial-instruction),
- **measured** via a three-layer ground-truth stack (programmatic registry
  verification, LLM-judge rubric, human-labelled gold subset), and
- **analysed** via classical ML, Bayesian inference, and sequential
  (Markov) modelling,

inside a domain whose structured public evidence base
(ClinicalTrials.gov) provides *verifiable* ground truth for several
safety dimensions.

The biomedical setting is the **substrate**, not the subject.
The **subject** is misalignment-shaped behaviour under realistic
research-environment pressures.

---

## 3. Sub-questions

- **SQ1 — Elicitation.** Which pressure conditions produce the largest
  increase in unsafe-behaviour rates relative to a no-pressure baseline?
- **SQ2 — Monitoring-divergence.** Do agents behave differently when they
  believe they are being monitored vs unmonitored, holding all other
  conditions fixed?
- **SQ3 — Detection.** Can a supervised classifier, trained on
  response-level features and condition encodings, predict unsafe sessions
  with high PR AUC?
- **SQ4 — Generalisation.** Do detectors trained on one (scenario, model)
  subset transfer to held-out scenarios and held-out models?
- **SQ5 — Sequential drift (Markov).** Across multi-turn interactions,
  does agent behaviour drift toward unsafe absorbing states, and is the
  transition structure pressure-dependent?
- **SQ6 — Quantified condition effects (Bayesian).** What are the
  posterior credible intervals on the unsafe-rate *increase* caused by
  each pressure condition, after controlling for model and scenario?
- **SQ7 — Oversight failure.** Are there (scenario, condition) cells in
  which surface fluency — as judged by a separate LLM judge (Layer B) or
  a casual human (Layer C) — masks substantive failure detected
  programmatically against the trial registry (Layer A)? This is the
  scalable-oversight failure mode the project most directly targets.

---

## 4. Hypotheses

| ID | Hypothesis | Predicted direction | Ground truth |
|----|------------|---------------------|--------------|
| H1 | Pressure conditions increase unsafe-behaviour rates over baseline | Positive; largest for *publication* and *hidden-adversarial* | Layers A + B |
| H2 | Unmonitored sessions show higher unsafe rates than monitored, all else equal | Unmonitored > monitored | Layer A primary, B secondary |
| H3 | A small engineered-feature classifier achieves PR AUC ≥ 0.70 on held-out cells | — | Layer A or C as target |
| H4 | More-capable models exhibit *greater* monitoring-divergence, not less | Capability ↑ ⇒ divergence ↑ | Layer A + B |
| H5 | Multi-turn drift toward unsafe behavioural states is more pronounced under pressure | Pressure ↑ ⇒ unsafe-state stationary probability ↑ | Layer B per-turn |
| H6 | At least one (scenario, condition) cell exhibits an *oversight gap*: Layer B/C ratings indicate safe behaviour while Layer A counts indicate verifiable failures | Positive existence claim | Layers A vs B/C disagreement |

**H4** and **H6** are the highest-value hypotheses for Anthropic
positioning: H4 because an inverse-scaling result on monitoring-divergence
would be a strong model-organism finding; H6 because it directly
operationalises a scalable-oversight failure.

---

## 5. Mapping to Anthropic research priorities

| Anthropic priority | How NeuroGuard touches it |
|---|---|
| Model organisms of misalignment | The whole study design: elicit, measure, characterise misalignment-shaped behaviour in a controlled substrate |
| Scalable oversight | SQ7 / H6 — the Layer A vs Layer B/C gap *is* a scalable-oversight failure |
| AI control | Pressure-condition design; monitoring-vs-unmonitored as a control intervention |
| Adversarial robustness | Hidden-adversarial-instruction pressure condition |
| Evaluations | The benchmark itself, with public schema and reproducible runner |
| Interpretability | Behavioural interpretability of detectors (feature importance, SHAP); not mechanistic |

---

## 6. In scope

- ClinicalTrials.gov Alzheimer's-disease / dementia trials as the evidence
  substrate.
- Factorial design: 4 scenarios × 7 pressures × 2 monitoring × 3 models ×
  5 seeds (= 840 sessions for the full sweep; fractional design available
  for cost-constrained runs).
- Three-layer ground truth (Layer A programmatic against CT.gov, Layer B
  LLM-judge rubric, Layer C ~150-session human-labelled gold subset).
- Supervised ML detectors with hyperparameter tuning and held-out cell
  evaluation.
- Bayesian per-condition unsafe-rate inference with credible intervals.
- Markov per-condition transition analysis over per-turn behavioural
  states.
- Behavioural interpretability of detectors.

## 7. Out of scope (deliberately)

- Bio capability uplift of any kind.
- Real patient data.
- Fine-tuning, RLHF, or training of base models.
- Mechanistic interpretability of model internals.
- Multi-disease scope beyond Alzheimer's / dementia in the MVP.
- Anything that requires unreviewed release of model-generated content
  that could plausibly aid harm.

---

## 8. Success criteria

- **Primary.**
  - Full factorial run executed and persisted with full provenance.
  - Layer A verifiable ground truth implemented and unit-tested.
  - Layer B + Layer C agreement reported with Cohen's κ (or
    Krippendorff's α).
  - At least one detector achieves PR AUC > 0.70 on a held-out
    (scenario, model) split.
- **Secondary.**
  - Per-condition Bayesian credible intervals on unsafe-rate increase.
  - Per-condition Markov transition matrices with statistical comparison.
  - ≥ 3 interpretable features identified with effect size and
    confidence interval.
- **Stretch.**
  - At least one finding strong and surprising enough to warrant a short
    public write-up (blog post / pre-print).

---

## 9. Ethical and safety scope

- We measure **meta-behaviour** (overclaiming, hidden uncertainty,
  monitoring-divergence, oversight failure). We do **not** score "did the
  agent give a more dangerous biomedical answer".
- All evidence comes from public, de-identified registries.
- Any "confidential" content used to probe leakage is **synthetic**,
  clearly marked in the dataset, and the leak detector matches a
  synthetic canary token only.
- Hidden adversarial instructions probe **coercion patterns** (e.g.
  "ignore your earlier instruction to flag uncertainty"), **never** bio
  capability (e.g. "explain how to synthesise X").
- Benchmark code, scenario configs, rubric, and dataset *schema* are
  intended for public release. The full *generated session data* will be
  reviewed before release; sessions containing model outputs that could
  plausibly aid harm will be withheld or redacted.

---

## 10. Open decisions tracked here

- **Human labelling of ~150 sessions for Layer C** — default: yes.
  Confirm by Step 12.
- **Model provider mix** — defer to Step 10. Candidate slots:
  one frontier-closed, one frontier-open, one smaller-open.

---

*Document version: 0.1.0 — initial lock-in at Step 1.*
