from typing import Literal, get_args

from drift_happens.utils.paths import DATA_DIR

DATASET_DIR = DATA_DIR

DatasetName = Literal["amazon_reviews_23", "arxiv", "imdb_faces", "yearbook"]

DATASET_NAMES: list[DatasetName] = list(get_args(DatasetName))
