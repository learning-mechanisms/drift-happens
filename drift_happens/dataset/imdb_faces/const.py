from drift_happens.dataset.const import DATASET_DIR

DOWNLOAD_LINK = (
    "https://e.pcloud.link/publink/show?code=XZdlzcZDajLEoqJOdmLzyEPVcoWdu54WEWy"
)

IMDB_TARGET_DIR = DATASET_DIR / "imdb_faces"
IMDB_TAR_FILE = IMDB_TARGET_DIR / "imdb.tar.gz"
IMDB_UNPACK_DIR = IMDB_TARGET_DIR / "raw"
IMDB_PREPROCESSED_DIR = IMDB_TARGET_DIR / "processed"

IMDB_TENSOR_DATASET_CACHE = IMDB_TARGET_DIR / "cache_tensor_dataset"
