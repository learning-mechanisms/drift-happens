from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from drift_happens.runtime.aggregate import (
    RESULTS_SCHEMA,
    aggregate_metric_ledgers,
    write_results_parquet,
)


def _write_ledger(run_dir: Path, phase: str, rows: list[dict]) -> None:
    ledger = run_dir / "metrics" / f"{phase}.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_aggregate_collects_rows_across_runs(tmp_path: Path) -> None:
    _write_ledger(
        tmp_path / "run_a",
        "eval",
        [
            {
                "experiment": "exp",
                "dataset": "arxiv",
                "trainer": "ffn_l",
                "seed": 1,
                "phase": "eval",
                "metric": "eval/auc_macro",
                "value": 0.81,
                "train_slice": "2000",
                "eval_slice": "2005",
                "run_identity": {"config_hash": "abc", "snapshot_sha256": "def"},
            }
        ],
    )
    _write_ledger(
        tmp_path / "run_b",
        "train",
        [
            {
                "experiment": "exp",
                "dataset": "yearbook",
                "trainer": "mlp",
                "seed": 2,
                "phase": "train",
                "metric": "train/slice_completed",
                "value": 1.0,
            }
        ],
    )

    frame = aggregate_metric_ledgers(tmp_path)

    assert frame.height == 2
    assert set(frame.columns) == set(RESULTS_SCHEMA)
    auc_row = frame.filter(frame["metric"] == "eval/auc_macro").to_dicts()[0]
    assert auc_row["value"] == 0.81
    assert auc_row["config_hash"] == "abc"
    assert auc_row["eval_slice"] == "2005"


def test_aggregate_empty_runs_root_is_typed_and_empty(tmp_path: Path) -> None:
    frame = aggregate_metric_ledgers(tmp_path)

    assert frame.height == 0
    assert set(frame.columns) == set(RESULTS_SCHEMA)


def test_write_results_parquet_round_trips(tmp_path: Path) -> None:
    _write_ledger(
        tmp_path / "run_a",
        "eval",
        [
            {
                "experiment": "exp",
                "dataset": "arxiv",
                "metric": "eval/auc_macro",
                "value": 0.5,
                "phase": "eval",
            }
        ],
    )
    output = tmp_path / "results.parquet"

    written = write_results_parquet(runs_root=tmp_path, output=output)

    assert written == output
    assert pl.read_parquet(output).height == 1


def test_aggregate_skips_torn_ledger_line(tmp_path: Path) -> None:
    _write_ledger(
        tmp_path / "run_a",
        "eval",
        [
            {
                "experiment": "exp",
                "metric": "eval/auc_macro",
                "value": 0.5,
                "run_identity": {},
            }
        ],
    )
    ledger = tmp_path / "run_a" / "metrics" / "eval.jsonl"
    with ledger.open("a") as handle:
        handle.write('{"experiment": "exp", "metric": "eval/au')

    frame = aggregate_metric_ledgers(tmp_path)

    assert frame.height == 1


def test_aggregate_skips_rows_with_malformed_identity_or_seed(tmp_path: Path) -> None:
    _write_ledger(
        tmp_path / "run_a",
        "eval",
        [
            {"experiment": "exp", "metric": "good", "value": 1.0, "seed": 3},
            {
                "experiment": "exp",
                "metric": "bad-identity",
                "value": 1.0,
                "run_identity": "not-an-object",
            },
            {"experiment": "exp", "metric": "bad-seed", "value": 1.0, "seed": "three"},
            {"experiment": "exp", "metric": "bool-seed", "value": 1.0, "seed": True},
        ],
    )

    frame = aggregate_metric_ledgers(tmp_path)

    assert frame.get_column("metric").to_list() == ["good"]
    assert frame.get_column("seed").to_list() == [3]


def test_aggregate_skips_rows_with_non_numeric_value_step_or_epoch(
    tmp_path: Path,
) -> None:
    _write_ledger(
        tmp_path / "run_a",
        "eval",
        [
            {"experiment": "exp", "metric": "good", "value": 1, "step": 2, "epoch": 0},
            {"experiment": "exp", "metric": "bad-value", "value": "high"},
            {"experiment": "exp", "metric": "bool-value", "value": True},
            {"experiment": "exp", "metric": "bad-step", "value": 1.0, "step": "two"},
            {"experiment": "exp", "metric": "bad-epoch", "value": 1.0, "epoch": 1.5},
        ],
    )

    frame = aggregate_metric_ledgers(tmp_path)

    assert frame.get_column("metric").to_list() == ["good"]
    assert frame.get_column("value").to_list() == [1.0]


def test_aggregate_recovers_after_a_torn_middle_line(tmp_path: Path) -> None:
    ledger = tmp_path / "run_a" / "metrics" / "eval.jsonl"
    ledger.parent.mkdir(parents=True)
    rows = [
        json.dumps({"experiment": "exp", "metric": "m1", "value": 1.0}),
        '{"experiment": "exp", "metr',
        "42",
        json.dumps({"experiment": "exp", "metric": "m2", "value": 2.0}),
    ]
    ledger.write_text("\n".join(rows) + "\n")

    frame = aggregate_metric_ledgers(tmp_path)

    assert frame.height == 2
    assert set(frame.get_column("metric").to_list()) == {"m1", "m2"}
