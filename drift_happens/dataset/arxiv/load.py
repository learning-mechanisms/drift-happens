import pandas as pd

from drift_happens.dataset.arxiv.const import ARXIV_PREPROCESSED_DF
from drift_happens.dataset.arxiv.scope import filter_arxiv_top_leaf_label_scope


def load_arxiv() -> pd.DataFrame:
    """Load the canonical strict top-leaf-label title+abstract conference scope."""
    return filter_arxiv_top_leaf_label_scope(pd.read_parquet(ARXIV_PREPROCESSED_DF))
