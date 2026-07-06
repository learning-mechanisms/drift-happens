"""Aggregate completed local seed runs into confidence summaries."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy import stats

from drift_happens.configs import ExperimentConfig
from drift_happens.runtime.completion_filter import local_seed_statuses_by_identity
from drift_happens.utils.ids import slugify
from drift_happens.utils.log import get_logger
from drift_happens.utils.paths import REPORTS_DIR
from drift_happens.utils.wandb_identity import build_run_identity

logger = get_logger()


def summarize_seeds(
    cfg: ExperimentConfig,
    *,
    source_path: Path,
    seeds: tuple[int, ...],
    out_dir: Path | None = None,
    runs_root: Path | None = None,
    metric: str | None = None,
    write_csv: bool = False,
    write_markdown: bool = False,
) -> Path:
    """Aggregate local ``results/summary.json`` files for matching completed seeds."""
    identity = build_run_identity(cfg, run_dir=Path("summary"), source_path=source_path)
    identities = {
        seed: build_run_identity(
            cfg.model_copy(update={"seed": seed}),
            run_dir=Path("summary"),
            source_path=source_path,
        )
        for seed in seeds
    }
    rows = local_seed_statuses_by_identity(identities, runs_root=runs_root)
    seed_results: list[dict[str, Any]] = []
    missing_or_failed: list[dict[str, Any]] = []

    for row in rows:
        if row.status != "ok" or row.run_dir is None:
            missing_or_failed.append(
                {
                    "eval": row.eval,
                    "seed": row.seed,
                    "status": row.status,
                    "train": row.train,
                    "run_dir": str(row.run_dir) if row.run_dir else None,
                }
            )
            continue
        summary_path = row.run_dir / "results" / "summary.json"
        summary = _load_summary(summary_path)
        if summary is None:
            # A completed run with an unreadable summary must not count as a
            # successful seed with empty metrics.
            logger.warning(
                f"Excluding seed {row.seed}: missing or corrupt summary at {summary_path}"
            )
            missing_or_failed.append(
                {
                    "eval": row.eval,
                    "seed": row.seed,
                    "status": "corrupt_summary",
                    "train": row.train,
                    "run_dir": str(row.run_dir),
                }
            )
            continue
        metrics = summary.get("metrics", {})
        if metric is not None and isinstance(metrics, dict):
            metrics = {
                key: value
                for key, value in metrics.items()
                if key == metric or str(key).endswith(f"/{metric}")
            }
        seed_results.append(
            {
                "metrics": metrics,
                "primary_metric": summary.get("primary_metric"),
                "primary_value": summary.get("primary_value"),
                "run_dir": str(row.run_dir),
                "seed": row.seed,
                "status": row.status,
            }
        )

    payload = {
        "failed_or_corrupt_count": len(missing_or_failed),
        "identity": identity.model_dump(mode="json"),
        "metric_filter": metric,
        "metric_summaries": _metric_summaries(seed_results),
        "missing_or_failed": missing_or_failed,
        "seeds": list(seeds),
        "source_path": str(source_path),
        "successful_seeds": [row["seed"] for row in seed_results],
    }
    report_dir = out_dir or REPORTS_DIR / "seeds"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{slugify(identity.source_identity)}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if write_csv:
        _write_csv(path.with_suffix(".csv"), payload)
    if write_markdown:
        _write_markdown(path.with_suffix(".md"), payload)
    return path


def _metric_summaries(
    seed_results: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    values_by_metric: dict[str, list[float]] = {}
    for result in seed_results:
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            values_by_metric.setdefault(str(key), []).append(float(value))

    summaries: dict[str, dict[str, float | int]] = {}
    for metric, values in sorted(values_by_metric.items()):
        n = len(values)
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if n > 1 else 0.0
        stderr = std / math.sqrt(n)
        # Student-t multiplier: the normal 1.96 is too tight for few seeds.
        half_width = float(stats.t.ppf(0.975, n - 1)) * stderr if n > 1 else 0.0
        summaries[metric] = {
            "ci95_high": mean + half_width,
            "ci95_low": mean - half_width,
            "count": n,
            "max": max(values),
            "mean": mean,
            "min": min(values),
            "std": std,
            "stderr": stderr,
        }
    return summaries


def _load_summary(path: Path) -> dict[str, Any] | None:
    """Parse a run summary, returning ``None`` when missing or corrupt."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "count",
                "mean",
                "std",
                "stderr",
                "ci95_low",
                "ci95_high",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        for metric, summary in payload["metric_summaries"].items():
            writer.writerow({"metric": metric, **summary})


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Seed Summary: {payload['identity']['source_identity']}",
        "",
        "| metric | count | mean | std | stderr | ci95_low | ci95_high | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric, summary in payload["metric_summaries"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    str(summary["count"]),
                    _fmt(summary["mean"]),
                    _fmt(summary["std"]),
                    _fmt(summary["stderr"]),
                    _fmt(summary["ci95_low"]),
                    _fmt(summary["ci95_high"]),
                    _fmt(summary["min"]),
                    _fmt(summary["max"]),
                ]
            )
            + " |"
        )
    if payload["missing_or_failed"]:
        lines.extend(["", "## Missing Or Failed", ""])
        for row in payload["missing_or_failed"]:
            lines.append(
                f"- seed {row['seed']}: {row['status']} "
                f"(train={row.get('train')}, eval={row.get('eval')})"
            )
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
