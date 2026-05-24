"""Dataset assembly — transforms raw session JSONL into analysis-ready Parquet.

Produces four structured datasets:
- D2a: Session-level records (one row per session)
- D2b: Turn-level records (for Markov extension, currently 1 turn per session)
- D2c: Labels (Layer A + Layer B scores consolidated)
- D2d: Engineered features for ML classification
"""

from neuroguard.dataset.assemble import assemble_dataset, load_sessions_df
from neuroguard.dataset.features import engineer_features

__all__ = [
    "assemble_dataset",
    "engineer_features",
    "load_sessions_df",
]
