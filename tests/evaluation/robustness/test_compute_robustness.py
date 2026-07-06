from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_happens.evaluation.robustness.compute_robustness import (
    ROBUSTNESS_FIELDS,
    SkippedEntry,
    compute_runtime_robustness_scores,
    write_robustness_reports,
)
from drift_happens.evaluation.robustness.metrics import analyze_drift_robustness

AGGREGATED_COLUMNS = ["model", "seeds"] + [
    f"{column}_{suffix}" for column in ROBUSTNESS_FIELDS for suffix in ("mean", "std")
]


def _write_run(
    root: Path,
    *,
    matrix: dict,
    dataset: str = "yearbook",
    trainer: str = "cnn_s",
    experiment: str = "cnn-small",
    seed: int = 0,
) -> None:
    run_dir = (
        root
        / dataset
        / trainer
        / experiment
        / f"seed={seed}"
        / f"{dataset}__{experiment}__cfg-abc"
    )
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "dataset": {"name": dataset},
                "name": experiment,
                "seed": seed,
                "trainer": {"key": trainer},
            }
        )
    )
    (run_dir / "results" / "drift_matrix.json").write_text(json.dumps(matrix))


def _accuracy_matrix(values: np.ndarray) -> dict:
    keys = [str(2000 + offset) for offset in range(values.shape[0])]
    return {
        train_key: {
            eval_key: {"accuracy": values[i, j]}
            for j, eval_key in enumerate(keys)
            if not np.isnan(values[i, j])
        }
        for i, train_key in enumerate(keys)
    }


def test_runtime_scores_aggregate_mean_and_std_across_two_seeds(
    tmp_path: Path,
) -> None:
    seed_matrices = {
        0: np.array([[0.9, 0.8], [np.nan, 0.85]]),
        1: np.array([[0.7, 0.6], [np.nan, 0.75]]),
    }
    for seed, values in seed_matrices.items():
        _write_run(tmp_path / "runs", matrix=_accuracy_matrix(values), seed=seed)

    frames = compute_runtime_robustness_scores(runs_root=tmp_path / "runs")

    assert list(frames) == ["yearbook"]
    df = frames["yearbook"]
    assert df.columns.tolist() == AGGREGATED_COLUMNS
    assert df["model"].tolist() == ["cnn_s"]
    assert df.loc[0, "seeds"] == 2
    expected = [analyze_drift_robustness(values) for values in seed_matrices.values()]
    strengths = [result.strength for result in expected]
    headlines = [result.combined_harmonic for result in expected]
    assert df.loc[0, "S_mean"] == pytest.approx(np.mean(strengths))
    assert df.loc[0, "S_std"] == pytest.approx(np.std(strengths, ddof=1))
    assert df.loc[0, "H_mean"] == pytest.approx(np.mean(headlines))
    assert df.loc[0, "H_std"] == pytest.approx(np.std(headlines, ddof=1))


def test_runtime_scores_use_dataset_primary_metric(tmp_path: Path) -> None:
    # auc_macro is arxiv's primary metric and must win over accuracy, even when
    # the runtime matrix stores it under the eval/ prefix.
    matrix = {
        "2010": {
            "2010": {"accuracy": 0.5, "eval/auc_macro": 0.9},
            "2011": {"accuracy": 0.5, "eval/auc_macro": 0.6},
        },
        "2011": {"2011": {"accuracy": 0.5, "eval/auc_macro": 0.8}},
    }
    _write_run(tmp_path / "runs", matrix=matrix, dataset="arxiv", trainer="bert_s")

    df = compute_runtime_robustness_scores(runs_root=tmp_path / "runs")["arxiv"]

    expected = analyze_drift_robustness(np.array([[0.9, 0.6], [np.nan, 0.8]]))
    assert df.loc[0, "S_mean"] == pytest.approx(expected.strength)
    assert df.loc[0, "R_rel_mean"] == pytest.approx(expected.robustness_rel)


def test_runtime_scores_treat_balanced_mse_as_lower_is_better(tmp_path: Path) -> None:
    matrix = {
        "0": {"0": {"balanced_mse": 0.2}, "1": {"balanced_mse": 0.5}},
        "1": {"1": {"balanced_mse": 0.4}},
    }
    _write_run(
        tmp_path / "runs",
        matrix=matrix,
        dataset="amazon_reviews_23",
        trainer="bert_s",
    )

    df = compute_runtime_robustness_scores(runs_root=tmp_path / "runs")[
        "amazon_reviews_23"
    ]

    expected = analyze_drift_robustness(
        np.array([[0.2, 0.5], [np.nan, 0.4]]), metric_type="lower_is_better"
    )
    assert df.loc[0, "S_mean"] == pytest.approx(expected.strength)
    assert df.loc[0, "H_mean"] == pytest.approx(expected.combined_harmonic)


def test_write_robustness_reports_ranks_models_and_writes_csv_per_dataset(
    tmp_path: Path,
) -> None:
    strong = np.array([[0.9, 0.85], [np.nan, 0.9]])
    weak = np.array([[0.9, 0.3], [np.nan, 0.9]])
    _write_run(tmp_path / "runs", matrix=_accuracy_matrix(strong), trainer="cnn_l")
    _write_run(tmp_path / "runs", matrix=_accuracy_matrix(weak), trainer="cnn_s")

    written = write_robustness_reports(
        runs_root=tmp_path / "runs", output_dir=tmp_path / "out"
    )

    assert list(written) == ["yearbook"]
    assert written["yearbook"] == tmp_path / "out" / "yearbook_robustness.csv"
    df = pd.read_csv(written["yearbook"])
    assert df.columns.tolist() == AGGREGATED_COLUMNS
    assert df["model"].tolist() == ["cnn_l", "cnn_s"]
    # Single-seed groups have zero std.
    assert df["H_std"].tolist() == [0.0, 0.0]


def test_runtime_skips_are_collected_and_summarized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = np.array([[0.9, 0.8], [np.nan, 0.85]])
    _write_run(tmp_path / "runs", matrix=_accuracy_matrix(good), trainer="cnn_l")
    # yearbook's primary metric is accuracy, so this run can only be skipped.
    _write_run(
        tmp_path / "runs", matrix={"2000": {"2000": {"f1": 0.5}}}, trainer="cnn_s"
    )

    written = write_robustness_reports(
        runs_root=tmp_path / "runs", output_dir=tmp_path / "out"
    )

    assert list(written) == ["yearbook"]
    assert pd.read_csv(written["yearbook"])["model"].tolist() == ["cnn_l"]
    out = capsys.readouterr().out
    assert "Skipped 1 entry:" in out
    assert "no 'accuracy' values on shared slices" in out


def test_strict_raises_when_anything_was_skipped(tmp_path: Path) -> None:
    _write_run(
        tmp_path / "runs", matrix={"2000": {"2000": {"f1": 0.5}}}, trainer="cnn_s"
    )

    with pytest.raises(RuntimeError, match="skipped 1 of the inputs"):
        write_robustness_reports(
            runs_root=tmp_path / "runs", output_dir=tmp_path / "out", strict=True
        )


def test_runtime_skips_matrix_with_nan_diagonal(tmp_path: Path) -> None:
    # A missing in-distribution baseline (NaN on the diagonal) poisons the
    # robustness scores, so the run must be skipped rather than scored as a
    # degenerate H=0.0 seed averaged into the per-trainer aggregate.
    skipped: list[SkippedEntry] = []
    _write_run(
        tmp_path / "runs",
        matrix=_accuracy_matrix(np.array([[np.nan, 0.8], [0.7, 0.85]])),
    )

    frames = compute_runtime_robustness_scores(
        runs_root=tmp_path / "runs", skipped=skipped
    )

    assert frames == {}
    assert len(skipped) == 1
    assert "shared slices" in skipped[0].reason


def test_runtime_skips_single_slice_matrix(tmp_path: Path) -> None:
    # A single shared slice has no future drift pair, so it cannot yield a
    # robustness score and must be skipped.
    skipped: list[SkippedEntry] = []
    _write_run(tmp_path / "runs", matrix=_accuracy_matrix(np.array([[0.9]])))

    frames = compute_runtime_robustness_scores(
        runs_root=tmp_path / "runs", skipped=skipped
    )

    assert frames == {}
    assert len(skipped) == 1
