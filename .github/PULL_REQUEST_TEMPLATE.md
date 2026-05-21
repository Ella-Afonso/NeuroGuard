## Summary

<!-- One paragraph: what changed and why. Link any related issue. -->

## Type of change

- [ ] Research design (scenarios, pressures, rubric, hypotheses)
- [ ] Pipeline / runner / data collection
- [ ] Analysis / ML / interpretability
- [ ] Tests
- [ ] CI / tooling / docs
- [ ] Other

## Scope-discipline check

- [ ] If this PR changes the **research question, hypotheses, scope, or
      success criteria**: I have updated `docs/research_question.md` in
      this same PR.
- [ ] If this PR changes the **safety rubric, scenarios, or pressure
      conditions**: I have bumped the relevant version field and
      documented the change.

## Engineering checks

- [ ] Tests added / updated
- [ ] `pytest` passes locally
- [ ] `ruff check` and `ruff format --check` clean
- [ ] `mypy` clean (or new errors documented)
- [ ] No secrets committed (`.env`, API keys, credentials)

## Reproducibility

- [ ] Random seeds set where new randomness was introduced
- [ ] Config files updated if defaults changed
- [ ] Provenance fields preserved (config hash, model version, seed)

## Safety / ethics

- [ ] No real patient data
- [ ] No bio capability uplift content
- [ ] Synthetic canary tokens only for leakage tests
- [ ] No release of model-generated content that could plausibly aid harm

## Notes for reviewer

<!-- Anything reviewer should focus on, known limitations, follow-ups. -->
