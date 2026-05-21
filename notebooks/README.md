# notebooks/

Exploratory notebooks. **Use sparingly.**

## Discipline rules

1. **Allowed here:** EDA, one-off prototyping, viewing model outputs
   interactively, sanity-checking results.

2. **Not allowed here:** anything another step, test, or the report
   depends on. If a notebook produces a number cited anywhere, the
   code that produces that number lives in `src/neuroguard/` and the
   notebook is a thin caller.

3. **Naming:** `NN_short_description.ipynb`
   (e.g. `01_evidence_eda.ipynb`, `02_session_eda.ipynb`).
   Numbering forces sequence.

4. **Cleared outputs:** `nbstripout` is configured as a pre-commit
   hook. Notebook diffs will only show code changes, never output
   blobs. Do not manually commit notebooks with outputs.

5. **Imports:** Always import from the installed package:
   ```python
   from neuroguard.features import extract_features
   from neuroguard.analysis import run_eda
   ```
   Never use `sys.path` hacks or relative imports.
