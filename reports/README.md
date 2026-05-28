# reports/

Paper-style write-up of NeuroGuard research findings.

## Contents

- `neuroguard.tex` (or `.typ`) — Main manuscript (~6–10 pages).
- `figures/` — Publication-quality figures referenced by the manuscript.
- `references.bib` — BibTeX references.

## Build instructions

```bash
# LaTeX
pdflatex neuroguard.tex && bibtex neuroguard && pdflatex neuroguard.tex && pdflatex neuroguard.tex

# Or Typst (recommended — faster, simpler)
typst compile neuroguard.typ
```

## Style

Target: NeurIPS / ICML workshop paper format.
Focus on: methods, experimental design, results tables, limitations.
