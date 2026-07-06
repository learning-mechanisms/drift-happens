from drift_happens.dataset.const import DATASET_DIR

ARXIV_TARGET_DIR = DATASET_DIR / "arxiv"
ARXIV_CACHE_DIR = ARXIV_TARGET_DIR / "cache"
ARXIV_CACHE_FILE = ARXIV_CACHE_DIR / "arxiv-metadata-oai-snapshot.json"
ARXIV_PREPROCESSED_DIR = ARXIV_TARGET_DIR / "processed"
ARXIV_PREPROCESSED_DF = ARXIV_PREPROCESSED_DIR / "arxiv.parquet"
