from dataclasses import dataclass

from drift_happens.dataset.const import DatasetName
from drift_happens.evaluation.robustness.metrics import MetricName
from drift_happens.utils.paths import ARTIFACTS_DIR


@dataclass
class DatasetExperimentConfig:
    metric: MetricName


DATASET_EXPERIMENT_DIRS: dict[DatasetName, DatasetExperimentConfig] = {
    "yearbook": DatasetExperimentConfig(metric="accuracy"),
    "imdb_faces": DatasetExperimentConfig(metric="accuracy"),
    "arxiv": DatasetExperimentConfig(metric="auc_macro"),
    "amazon_reviews_23": DatasetExperimentConfig(metric="balanced_mse"),
}

ROBUSTNESS_RESULTS_DIR = ARTIFACTS_DIR / "robustness_results"
