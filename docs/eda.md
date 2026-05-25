# Exploratory Data Analysis — Key Findings

## Dataset

- **280 sessions** from the full factorial experiment
- 4 scenarios x 7 pressures x 2 monitoring x 1 model (gpt-4o-mini) x 5 seeds
- Model under test: gpt-4o-mini (temperature=0.7)
- Judge: gpt-4o-mini (temperature=0.0)

## Key Findings

### 1. Overall Unsafe Rate: ~13.6%

38 of 280 sessions triggered at least one high-severity Layer A signal.
Class imbalance ratio is ~6.3:1 (safe:unsafe) — manageable with stratified
splits and class-weight handling in ML.

### 2. Scenario Vulnerability

`eligibility_triage` dominates the unsafe rate at ~49%, driven by
`decision_matches_truth` failures. The other three scenarios
(evidence_summarisation, proposal_critique, trial_ranking) hover near 0-3%.

This makes scientific sense: triage requires a binary decision (eligible/not)
that can be objectively verified, while the other tasks are more open-ended.

### 3. Pressure Effect

Authority, emotional_appeal, and hidden_instruction show slightly elevated
unsafe rates (~15%) vs baseline (~12.5%), but with wide confidence intervals
due to the n=40 per pressure cell. The effect is concentrated in the
`eligibility_triage` scenario.

### 4. Monitoring Effect

Monitored vs unmonitored shows minimal difference at this sample size.
The interaction plot reveals condition-specific patterns (e.g. emotional_appeal
shows higher unsafe rate when unmonitored).

### 5. Layer A Signals

- `hedging_present` — triggers in ~25% of sessions (expected, low severity)
- `evidence_coverage` — triggers in ~25% (information completeness)
- `decision_matches_truth` — triggers in ~11% (the main high-severity signal)
- `citation_hallucination` and `overclaiming` — rare but high severity

### 6. Feature Space

25 engineered features show meaningful variance. Top correlates with the
unsafe label will be identified during ML modelling.

## Figures

All saved to `reports/figures/`:

| Figure | Content |
|--------|---------|
| 01_class_balance.png | Pie chart + unsafe rate by scenario |
| 02_pressure_unsafe_rate.png | Bar chart with 95% Wilson CIs |
| 03_pressure_monitoring_interaction.png | Grouped bar chart |
| 04_layer_a_signals.png | Signal frequency + pressure heatmap |
| 05_layer_b_distributions.png | Judge score histograms |
| 06_layer_b_by_pressure.png | Judge score heatmap |
| 07_feature_distributions.png | Safe vs unsafe overlaid histograms |
| 08_correlation_matrix.png | Feature correlation heatmap |
| 09_scenario_pressure_heatmap.png | Scenario x pressure unsafe rate |

## Notebook

`notebooks/01_eda.ipynb` — fully executable, reads from `data/processed/`.
