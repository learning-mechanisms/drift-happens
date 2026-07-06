from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import TypeAdapter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from drift_happens.evaluation.metrics import (
    ClassificationMetrics,
    ClassificationMetricsUnion,
    MultiLabelClassificationMetrics,
    MultiLabelROCAUCTracker,
    RegressiveClassificationMetrics,
)


def test_classification_metrics_binary_confusion_loss_and_metrics() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.8, 0.2]])

    metrics = ClassificationMetrics.from_predictions(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 1, 0, 0]),
        y_prob=probs,
    )

    np.testing.assert_array_equal(metrics.confusion_matrix, [[2, 0], [1, 1]])
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.metrics["precision"] == pytest.approx(1.0)
    assert metrics.metrics["recall"] == pytest.approx(0.5)
    assert metrics.metrics["f1_score"] == pytest.approx(2 / 3)
    assert metrics.precision_balanced == pytest.approx(5 / 6)
    assert metrics.recall_balanced == pytest.approx(0.75)
    assert metrics.f1_score_macro == pytest.approx((0.8 + 2 / 3) / 2)
    assert metrics.loss is not None and np.isfinite(metrics.loss)


def test_balanced_accuracy_matches_sklearn() -> None:
    # balanced_accuracy is macro recall, i.e. scikit-learn's balanced accuracy.
    # The 400-sample draw covers all five classes, so the two agree exactly; they
    # differ only when a class is missing from y_true (our code scores it as zero
    # recall, sklearn drops it).
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=400)
    y_pred = rng.integers(0, 5, size=400)

    metrics = ClassificationMetrics.from_predictions(y_true, y_pred, num_classes=5)

    assert metrics.balanced_accuracy == pytest.approx(
        balanced_accuracy_score(y_true, y_pred)
    )
    assert metrics.balanced_accuracy == pytest.approx(metrics.recall_balanced)
    assert "balanced_accuracy" in metrics.metrics
    assert "accuracy_balanced" not in metrics.metrics


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_multiclass_metrics_match_sklearn(seed: int) -> None:
    # accuracy and the macro precision/recall/F1 reproduce scikit-learn exactly.
    # labels=range(5) pins the macro denominator to all five classes, matching our
    # code, so the two agree even on a draw that happens to miss a class.
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 5, size=400)
    y_pred = rng.integers(0, 5, size=400)
    labels = list(range(5))

    metrics = ClassificationMetrics.from_predictions(y_true, y_pred, num_classes=5)

    assert metrics.accuracy == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics.precision_balanced == pytest.approx(
        precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    assert metrics.recall_balanced == pytest.approx(
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    assert metrics.f1_score_macro == pytest.approx(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_multilabel_metrics_match_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(400, 3))
    y_pred = rng.integers(0, 2, size=(400, 3))

    metrics = MultiLabelClassificationMetrics.from_predictions(3, y_true, y_pred)

    assert metrics.f1_scores_macro == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    assert float(np.mean(list(metrics.precisions.values()))) == pytest.approx(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    assert float(np.mean(list(metrics.recalls.values()))) == pytest.approx(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )

    sk_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    sk_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    sk_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for c in range(3):
        assert metrics.precisions[c] == pytest.approx(sk_precision[c])
        assert metrics.recalls[c] == pytest.approx(sk_recall[c])
        assert metrics.f1_scores[c] == pytest.approx(sk_f1[c])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_multiclass_per_class_metrics_match_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 5, size=400)
    y_pred = rng.integers(0, 5, size=400)
    labels = list(range(5))

    metrics = ClassificationMetrics.from_predictions(y_true, y_pred, num_classes=5)

    sk_precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    sk_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    sk_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for c in labels:
        assert metrics.precision_per_class[c] == pytest.approx(sk_precision[c])
        assert metrics.recall_per_class[c] == pytest.approx(sk_recall[c])
        assert metrics.f1_score_per_class[c] == pytest.approx(sk_f1[c])


def test_metrics_match_sklearn_when_a_class_is_never_predicted() -> None:
    # Class 2 is never predicted, so its precision is undefined: our code and
    # scikit-learn (zero_division=0) both score it 0, in the per-class values and
    # in the macro means.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 1])
    labels = [0, 1, 2]

    metrics = ClassificationMetrics.from_predictions(y_true, y_pred, num_classes=3)

    sk_precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    for c in labels:
        assert metrics.precision_per_class[c] == pytest.approx(sk_precision[c])
    assert metrics.precision_balanced == pytest.approx(
        precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    assert metrics.recall_balanced == pytest.approx(
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    assert metrics.f1_score_macro == pytest.approx(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_roc_auc_tracker_matches_sklearn(seed: int) -> None:
    # The tracker computes the exact rank-based AUC, so it equals scikit-learn's.
    rng = np.random.default_rng(seed)
    y_prob = rng.uniform(0.0, 1.0, size=(400, 2))
    y_true = (rng.uniform(size=(400, 2)) < y_prob).astype(int)

    tracker = MultiLabelROCAUCTracker.from_predictions(
        2, y_true, np.zeros_like(y_true), y_prob
    )

    assert tracker.auc_macro == pytest.approx(
        roc_auc_score(y_true, y_prob, average="macro")
    )


def test_roc_auc_tracker_matches_sklearn_for_tightly_clustered_scores() -> None:
    # The negatives all score just above the positives, so the labels are perfectly
    # anti-ranked and the exact AUC is 0.0 — the rank order is preserved even though
    # the scores span a range narrower than any reasonable histogram bin.
    positives = np.linspace(0.501, 0.504, 50)
    negatives = np.linspace(0.505, 0.508, 50)
    y_prob = np.concatenate([positives, negatives]).reshape(-1, 1)
    y_true = np.concatenate([np.ones(50), np.zeros(50)]).astype(int).reshape(-1, 1)

    tracker = MultiLabelROCAUCTracker.from_predictions(
        1, y_true, np.zeros_like(y_true), y_prob
    )

    assert tracker.auc_macro == pytest.approx(roc_auc_score(y_true[:, 0], y_prob[:, 0]))
    assert tracker.auc_macro == pytest.approx(0.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_regression_metrics_match_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    actual = rng.integers(1, 6, size=200)
    predicted = actual + rng.normal(0.0, 1.0, size=200)

    metrics = RegressiveClassificationMetrics.from_predictions(actual, predicted)

    assert metrics.mae == pytest.approx(mean_absolute_error(actual, predicted))
    assert metrics.mse == pytest.approx(mean_squared_error(actual, predicted))
    assert metrics.rmse == pytest.approx(root_mean_squared_error(actual, predicted))
    assert metrics.r2_score == pytest.approx(r2_score(actual, predicted))


def test_balanced_mse_equals_plain_mse_for_equal_class_sizes() -> None:
    # With equal-sized classes the per-class average collapses to the overall MSE; the
    # two diverge only when the classes are imbalanced (see
    # test_regressive_classification_metrics_values).
    rng = np.random.default_rng(0)
    actual = np.array([1, 2, 3, 4, 5] * 40)
    predicted = actual + rng.normal(0.0, 1.0, size=actual.size)

    metrics = RegressiveClassificationMetrics.from_predictions(actual, predicted)

    assert metrics.balanced_mse == pytest.approx(mean_squared_error(actual, predicted))


def test_regression_metrics_are_nan_without_predictions() -> None:
    metrics = RegressiveClassificationMetrics.from_predictions(np.array([1, 2, 3]))

    assert np.isnan(metrics.mae)
    assert np.isnan(metrics.mse)
    assert np.isnan(metrics.rmse)
    assert np.isnan(metrics.r2_score)
    assert np.isnan(metrics.balanced_mse)


def test_classification_metrics_empty_input_uses_requested_class_count() -> None:
    metrics = ClassificationMetrics.from_predictions(
        np.array([], dtype=int), np.array([], dtype=int), num_classes=2
    )

    np.testing.assert_array_equal(metrics.confusion_matrix, np.zeros((2, 2)))
    assert all(value == 0.0 for value in metrics.metrics.values())


def test_classification_metrics_keeps_missing_classes_when_count_is_explicit() -> None:
    metrics = ClassificationMetrics.from_predictions(
        np.array([0, 0]), np.array([0, 0]), num_classes=3
    )

    np.testing.assert_array_equal(
        metrics.confusion_matrix,
        [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
    )


def test_classification_metrics_rejects_loss_and_loss_func_together() -> None:
    with pytest.raises(ValueError, match="either loss and loss_func"):
        ClassificationMetrics.from_predictions(
            np.array([0]), np.array([0]), loss=0.1, loss_func="cross_entropy"
        )


def test_classification_metrics_rejects_requested_loss_func_without_probs() -> None:
    # An explicit loss_func request that cannot be honored (no y_prob) must fail
    # loudly instead of silently dropping the loss.
    with pytest.raises(ValueError, match="loss_func was requested"):
        ClassificationMetrics.from_predictions(
            np.array([0, 1]), np.array([0, 1]), loss_func="cross_entropy"
        )


def test_classification_metrics_empty_input_nulls_loss_func_without_loss() -> None:
    # An empty batch computes no loss, so it must not advertise a loss_func,
    # matching the non-empty no-probability path.
    metrics = ClassificationMetrics.from_predictions(
        np.array([], dtype=int), np.array([], dtype=int), num_classes=2
    )

    assert metrics.loss is None
    assert metrics.loss_func is None


def test_classification_metrics_handles_non_contiguous_labels() -> None:
    metrics = ClassificationMetrics.from_predictions(
        np.array([1, 2]), np.array([1, 2]), num_classes=3
    )

    np.testing.assert_array_equal(
        metrics.confusion_matrix,
        [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
    )


def test_classification_metrics_rejects_labels_outside_explicit_class_count() -> None:
    with pytest.raises(ValueError, match="outside num_classes"):
        ClassificationMetrics.from_predictions(
            np.array([0, 2]), np.array([0, 1]), num_classes=2
        )


def test_multilabel_classification_metrics_per_class_values() -> None:
    y_true = np.array([[1, 0, 1], [0, 1, 1]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1]])

    metrics = MultiLabelClassificationMetrics.from_predictions(3, y_true, y_pred)

    np.testing.assert_array_equal(metrics.confusion_matrices[0], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(metrics.confusion_matrices[2], [[0, 0], [1, 1]])
    assert metrics.f1_scores[0] == pytest.approx(1.0)
    assert metrics.f1_scores[2] == pytest.approx(2 / 3)
    assert metrics.f1_scores_macro == pytest.approx(8 / 9)


def test_multilabel_roc_auc_tracker_handles_missing_classes() -> None:
    y_true = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]])
    y_prob = np.array([[0.9, 0.2, 0.7], [0.1, 0.3, 0.8], [0.8, 0.4, 0.9]])

    tracker = MultiLabelROCAUCTracker.from_predictions(
        3, y_true, np.zeros_like(y_true), y_prob
    )
    auc = tracker.auc

    assert np.isfinite(auc["per_class"][0])
    assert np.isnan(auc["per_class"][1])
    assert np.isnan(auc["per_class"][2])
    # Class 1 has no positive samples; class 2 has no negative samples — each lacks
    # one label, so AUC is undefined.  The macro averages only finite-AUC classes.
    assert auc["macro"] == pytest.approx(auc["per_class"][0])


def test_regressive_classification_metrics_values() -> None:
    metrics = RegressiveClassificationMetrics.from_predictions(
        np.array([1, 2, 2, 3]), predicted=np.array([1.5, 2.0, 1.0, 4.0])
    )

    assert metrics.mae == pytest.approx(0.625)
    assert metrics.mse == pytest.approx(0.5625)
    assert metrics.rmse == pytest.approx(0.75)
    assert metrics.r2_score == pytest.approx(-0.125)
    assert metrics.balanced_mse == pytest.approx((0.25 + 0.5 + 1.0) / 3)
    assert np.isnan(RegressiveClassificationMetrics.from_predictions(np.array([1])).mae)


def _is_finite_float_mapping(values: dict) -> bool:
    return all(
        isinstance(v, float) and not isinstance(v, bool) and np.isfinite(v)
        for v in values.values()
    )


def test_scalar_metrics_classification_includes_loss() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.8, 0.2]])
    metrics = ClassificationMetrics.from_predictions(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 1, 0, 0]),
        y_prob=probs,
    )

    values = metrics.scalar_metrics()

    assert _is_finite_float_mapping(values)
    assert values["accuracy"] == pytest.approx(0.75)
    assert values["f1_score_macro"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert "loss" in values


def test_scalar_metrics_regressive_exposes_regression_scalars() -> None:
    metrics = RegressiveClassificationMetrics.from_predictions(
        np.array([1, 2, 2, 3]), predicted=np.array([1.5, 2.0, 1.0, 4.0])
    )

    values = metrics.scalar_metrics()

    assert _is_finite_float_mapping(values)
    assert values["mse"] == pytest.approx(0.5625)
    assert values["rmse"] == pytest.approx(0.75)


def test_scalar_metrics_regressive_drops_nan_for_empty_predictions() -> None:
    metrics = RegressiveClassificationMetrics.from_predictions(np.array([1]))

    values = metrics.scalar_metrics()

    assert values == {}


def test_scalar_metrics_multilabel_classification_exposes_macro_and_per_class() -> None:
    y_true = np.array([[1, 0, 1], [0, 1, 1]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1]])
    metrics = MultiLabelClassificationMetrics.from_predictions(3, y_true, y_pred)

    values = metrics.scalar_metrics()

    assert _is_finite_float_mapping(values)
    assert values["f1_score_macro"] == pytest.approx(8 / 9)
    assert values["f1_score_class_0"] == pytest.approx(1.0)
    assert "accuracy_class_2" in values


def test_scalar_metrics_roc_auc_tracker_emits_macro_and_drops_nan_classes() -> None:
    y_true = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]])
    y_prob = np.array([[0.9, 0.2, 0.7], [0.1, 0.3, 0.8], [0.8, 0.4, 0.9]])
    tracker = MultiLabelROCAUCTracker.from_predictions(
        3, y_true, np.zeros_like(y_true), y_prob
    )

    values = tracker.scalar_metrics()

    assert _is_finite_float_mapping(values)
    assert "auc_macro" in values
    # Class 1 has no positive samples; class 2 has no negative samples -> nan -> dropped.
    assert "auc_class_0" in values
    assert "auc_class_1" not in values
    assert "auc_class_2" not in values


def test_roc_auc_tracker_round_trips_through_the_metrics_union() -> None:
    # The serialized tracker must deserialize back to its own class through the
    # untagged union, so the drift-matrix readers recover the same macro AUC.
    y_true = np.array([[1, 0], [0, 0], [1, 0], [0, 1]])
    y_prob = np.array([[0.9, 0.2], [0.1, 0.3], [0.8, 0.4], [0.3, 0.7]])
    tracker = MultiLabelROCAUCTracker.from_predictions(
        2, y_true, np.zeros_like(y_true), y_prob
    )

    adapter = TypeAdapter(ClassificationMetricsUnion)
    restored = adapter.validate_json(tracker.model_dump_json())

    assert isinstance(restored, MultiLabelROCAUCTracker)
    assert restored.auc_macro == pytest.approx(tracker.auc_macro)


def test_roc_auc_tracker_empty_slice_round_trips_without_crashing() -> None:
    tracker = MultiLabelROCAUCTracker.from_predictions(
        3, np.zeros((0, 3)), np.array([]), np.array([])
    )

    restored = MultiLabelROCAUCTracker.model_validate_json(tracker.model_dump_json())

    assert tracker.scalar_metrics() == {}
    assert restored.scalar_metrics() == {}
    assert np.isnan(restored.auc_macro)


def test_roc_auc_tracker_json_omits_raw_arrays_but_keeps_scalars() -> None:
    # The on-disk cell must stay tiny: only the scalar AUCs, never the per-row
    # labels/scores matrices that would balloon to gigabytes at scale.
    y_true = np.array([[1, 0], [0, 0], [1, 0], [0, 1]])
    y_prob = np.array([[0.9, 0.2], [0.1, 0.3], [0.8, 0.4], [0.3, 0.7]])
    tracker = MultiLabelROCAUCTracker.from_predictions(
        2, y_true, np.zeros_like(y_true), y_prob
    )

    payload = json.loads(tracker.model_dump_json())

    assert "labels" not in payload
    assert "scores" not in payload
    assert payload["auc_per_class"] == pytest.approx(tracker.auc["per_class"])

    restored = MultiLabelROCAUCTracker.model_validate_json(tracker.model_dump_json())
    assert restored.scalar_metrics() == pytest.approx(tracker.scalar_metrics())


def test_regression_metrics_json_omits_raw_vectors_but_keeps_scalars() -> None:
    actual = np.array([5, 5, 5, 1])
    predicted = np.array([5.0, 4.0, 5.0, 3.0])
    metrics = RegressiveClassificationMetrics.from_predictions(actual, predicted)

    payload = json.loads(metrics.model_dump_json())

    assert "actual" not in payload
    assert "predicted" not in payload
    assert payload["scalar_summary"]["balanced_mse"] == pytest.approx(
        metrics.balanced_mse
    )

    adapter = TypeAdapter(ClassificationMetricsUnion)
    restored = adapter.validate_json(metrics.model_dump_json())
    assert isinstance(restored, RegressiveClassificationMetrics)
    assert restored.scalar_metrics() == pytest.approx(metrics.scalar_metrics())


def test_legacy_metric_artifacts_still_load_through_the_union() -> None:
    # Older artifacts embedded the raw arrays without the scalar summaries; the
    # union must still route them to the right class and recover the metric.
    adapter = TypeAdapter(ClassificationMetricsUnion)

    legacy_auc = adapter.validate_json(
        '{"labels": [[1, 0], [0, 1]], "scores": [[0.9, 0.1], [0.2, 0.8]]}'
    )
    assert isinstance(legacy_auc, MultiLabelROCAUCTracker)
    assert legacy_auc.auc_macro == pytest.approx(1.0)

    legacy_reg = adapter.validate_json(
        '{"actual": [5, 5, 5, 1], "predicted": [5.0, 4.0, 5.0, 3.0]}'
    )
    assert isinstance(legacy_reg, RegressiveClassificationMetrics)
    assert legacy_reg.balanced_mse == pytest.approx(
        RegressiveClassificationMetrics.from_predictions(
            np.array([5, 5, 5, 1]), np.array([5.0, 4.0, 5.0, 3.0])
        ).balanced_mse
    )
