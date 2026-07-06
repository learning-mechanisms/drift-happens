import polars as pl

from drift_happens.dataset.amazon_reviews_23.const import (
    AR23_PREPROCESSED_REVIEWS_MERGED_PATH,
)
from drift_happens.dataset.amazon_reviews_23.scope import (
    build_amazon_reviews_23_scope,
)


def load_amazon_reviews_23() -> pl.DataFrame:
    """
    Load the conference scope: a deterministic stratified sample of the merged Amazon
    Reviews 2023 dump, up to 1M rows.

    The result reflects whatever is in the merged parquet; if that dump holds fewer in-
    scope rows than the sample target, the whole (smaller) frame is returned. Build the
    merged dump from the canonical group set with the dataset CLI to obtain the full
    1M-row sample.
    """
    if not AR23_PREPROCESSED_REVIEWS_MERGED_PATH.exists():
        raise FileNotFoundError(
            f"Merged reviews parquet not found at {AR23_PREPROCESSED_REVIEWS_MERGED_PATH}. "
            "Run `drift datasets-setup amazon-reviews-23 full` to build it."
        )
    return build_amazon_reviews_23_scope(
        pl.read_parquet(AR23_PREPROCESSED_REVIEWS_MERGED_PATH)
    )
