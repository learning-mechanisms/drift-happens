from drift_happens.dataset.const import DATASET_DIR

DOWNLOAD_LINK = (
    "https://e.pcloud.link/publink/show?code=XZ6lzcZJymgTsmpuDk25aekFFPR2fL7VPfV"
)

YB_TARGET_DIR = DATASET_DIR / "yearbook"
YB_TAR_FILE = YB_TARGET_DIR / "yearbook.tar.gz"
YB_UNPACK_DIR = YB_TARGET_DIR / "raw"
YB_PREPROCESSED_DIR = YB_TARGET_DIR / "processed"

YB_TENSOR_DATASET_CACHE = YB_TARGET_DIR / "cache_tensor_dataset"
