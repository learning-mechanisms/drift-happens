import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from drift_happens.dataset.amazon_reviews_23.const import (
    AR23_CACHE_DIR,
    AR23_PREPROCESSED_REVIEWS_DIR,
    AR23_PREPROCESSED_REVIEWS_MERGED_PATH,
)
from drift_happens.tools.cli import prompt_confirmation
from drift_happens.utils.log import get_logger

logger = get_logger()

# rows per incremental parquet write while streaming a review group
REVIEWS_CHUNK_ROWS = 100_000


def prepare_reviews_df(groups: list[str], skip_existing: bool = True) -> None:
    """Extract timestamp, category, title, text, rating."""
    groups = list(sorted(set(groups)))
    # pyarrow will not create missing parents, so make the output directory
    # before the ParquetWriter opens it.
    AR23_PREPROCESSED_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    for group in groups:
        parquet_path = AR23_PREPROCESSED_REVIEWS_DIR / f"{group}.parquet"
        if skip_existing and parquet_path.exists():
            logger.info(
                f"Preprocessed reviews for group {group} already exist, skipping..."
            )
            continue

        logger.info(f"Loading group: {group}")
        features = ["timestamp", "title", "text", "rating"]
        dataset_reviews = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            # Prefixes:
            # - raw_review_
            # - 0core_rating_only_
            # - 5core_rating_only_
            # - 0core_last_out_
            # - core_last_out_w_his_
            # - 0core_timestamp_
            # - 5core_timestamp_w_his_
            f"raw_review_{group}",
            split="full",
            cache_dir=AR23_CACHE_DIR,
            trust_remote_code=True,
            # too large to keep in memory
            streaming=True,
        )
        logger.info(f"Saving preprocessed reviews for group: {group}")
        tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        writer: pq.ParquetWriter | None = None
        schema: pa.Schema | None = None
        total_rows = 0
        try:
            for batch in dataset_reviews.iter(batch_size=REVIEWS_CHUNK_ROWS):
                chunk = pd.DataFrame({feature: batch[feature] for feature in features})
                chunk["rating"] = chunk["rating"].astype(int)
                chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="ms")

                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(tmp_path, schema, compression="zstd")
                else:
                    table = table.cast(schema)
                writer.write_table(table)
                total_rows += len(chunk)
        except BaseException:
            if writer is not None:
                writer.close()
            tmp_path.unlink(missing_ok=True)
            raise
        else:
            if writer is not None:
                writer.close()

        if writer is None:
            raise ValueError(f"No reviews streamed for group {group}")
        tmp_path.replace(parquet_path)

        logger.info(f"Loaded {total_rows} reviews for group: {group}")


def merge_review_dfs(groups: list[str], assume_yes: bool = False) -> None:
    """Merge preprocessed review dataframes for the specified groups."""
    logger.info("Merging preprocessed review dataframes...")

    if AR23_PREPROCESSED_REVIEWS_MERGED_PATH.exists() and not prompt_confirmation(
        f"Merged preprocessed reviews already exist at {AR23_PREPROCESSED_REVIEWS_MERGED_PATH}. Overwrite?",
        assume_yes=assume_yes,
    ):
        logger.info("Aborting merge.")
        return

    lazy_dfs = []
    for group in groups:
        parquet_path = AR23_PREPROCESSED_REVIEWS_DIR / f"{group}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Preprocessed reviews for group {group} do not exist. "
                "Please run `prepare_reviews_df` first."
            )
        lazy_df = (
            pl.scan_parquet(parquet_path)
            .select(["timestamp", "title", "text", "rating"])
            # category is implicit in the file name, stamp it from the group
            .with_columns(pl.lit(group).alias("category"))
        )
        lazy_dfs.append(lazy_df)

    merged_lazy_df = pl.concat(lazy_dfs)
    # Write to a temp sibling and atomically swap, so an interrupted merge never
    # leaves a corrupt parquet the existence check would treat as finished.
    tmp_path = AR23_PREPROCESSED_REVIEWS_MERGED_PATH.with_suffix(
        AR23_PREPROCESSED_REVIEWS_MERGED_PATH.suffix + ".tmp"
    )
    try:
        merged_lazy_df.sink_parquet(tmp_path, compression="zstd")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(AR23_PREPROCESSED_REVIEWS_MERGED_PATH)
