from pydantic import BaseModel, Field

from drift_happens.evaluation.metrics import ClassificationMetricsUnion


class TrainerEvaluationResults(BaseModel):
    results: dict[str, ClassificationMetricsUnion] = Field(default_factory=dict)
    """Evaluation results keyed by the stringified evaluation time-slice key."""
