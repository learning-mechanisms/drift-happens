"""Drift matrices, roster manifest, and result tables exported as JSON for the
website."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from drift_happens.analysis.datasets import DATASETS, DatasetSpec
from drift_happens.analysis.datasets.locations import DEFAULT_SITE_DIR
from drift_happens.analysis.plots import derive
from drift_happens.analysis.plots.names import (
    FAMILY_LABELS,
    figure_name,
    get_display_name,
    slice_label,
)

# Periods along each axis of the compact replay matrices (the hero player).
REPLAY_PERIODS = 12


def build_site_data(
    frame: pl.DataFrame,
    site_dir: Path = DEFAULT_SITE_DIR,
    expected: dict[str, list[str]] | None = None,
) -> list[Path]:
    """
    Write per-dataset matrices, roster manifest, and result tables as JSON.

    Each dataset's cohort mean lands at ``data/<slug>.json``; every model with data
    (complete or partial) at ``data/<slug>/<Model>.json``; the planned roster with run
    status at ``data/<slug>/manifest.json``; and the paper's tables at
    ``data/<slug>/tables.json``. Missing models carry no matrix file and render blank.
    """
    data_dir = site_dir / "data"
    outputs: list[Path] = []
    replay: list[dict] = []
    for dataset, spec in DATASETS.items():
        models, _ = derive.per_model_matrices(frame, dataset)
        if not models:
            continue
        mean = derive.mean_over_models(models)
        raw_range = derive.raw_extent([mean, *models])
        outputs.append(
            _matrix(
                data_dir / f"{spec.slug}.json",
                spec,
                mean.slices,
                mean.mean,
                raw_range=raw_range,
            )
        )
        replay.append(_replay_dataset(spec, mean))

        lineup = derive.lineup_matrices(
            frame, dataset, expected.get(dataset) if expected else None
        )
        for entry in lineup:
            if entry.status is derive.Status.MISSING:
                continue
            outputs.append(
                _matrix(
                    data_dir / spec.slug / f"{figure_name(entry.matrix.model)}.json",
                    spec,
                    entry.matrix.slices,
                    entry.matrix.mean,
                    model=get_display_name(entry.matrix.model),
                    raw_range=raw_range,
                )
            )
        deviation_span = derive.deviation_extent(
            derive.deviations([entry.matrix for entry in lineup], mean)
        )
        outputs.append(
            _manifest(
                data_dir / spec.slug / "manifest.json",
                dataset,
                spec,
                lineup,
                deviation_span,
            )
        )
        outputs.append(
            _tables(data_dir / spec.slug / "tables.json", dataset, spec, models, frame)
        )
    if replay:
        outputs.append(_replay(data_dir / "replay.json", replay))
    roster = _rosters(data_dir / "rosters.json")
    if roster is not None:
        outputs.append(roster)
    return outputs


# Site-friendly relabels and presentation strings for the model roster export.
_ROSTER_FAMILY = {"TX": "Transformer", "BiLSTM-Attn": "BiLSTM + attn"}
_ROSTER_ENCODER = {
    "ResNet50-IN": "ResNet-50",
    "ViT-S16-IN21k": "ViT-S/16",
    "EVA02-B": "EVA-02-B",
    "CLIP-B32": "CLIP-B/32",
}
_ROSTER_FAMILY_ORDER = {
    "MLP": 0,
    "CNN": 1,
    "ResNet": 2,
    "ViT": 3,
    "FFN": 0,
    "TextCNN": 1,
    "BiGRU": 2,
    "BiLSTM": 3,
    "BiLSTM + attn": 4,
    "Transformer": 5,
}
_ROSTER_SIZE_ORDER = {"S": 0, "M": 1, "L": 2}
_ROSTER_BIAS = {
    "MLP": "pixels as a flat vector, no spatial structure",
    "CNN": "locality and translation invariance",
    "ResNet": "convolution with skip connections",
    "ViT": "image patches compared by self-attention",
    "FFN": "ignores word order (mean-pooled embeddings)",
    "TextCNN": "local n-gram features",
    "BiGRU": "reads tokens in sequence",
    "BiLSTM": "reads tokens in sequence",
    "BiLSTM + attn": "sequence with attention pooling",
    "Transformer": "self-attention over all positions",
}
_ROSTER_PRETRAIN = {
    "vit_s16_in21k_frozen": "supervised, ImageNet-21k",
    "resnet50_in_frozen": "supervised, ImageNet-1k",
    "convnext_s_frozen": "supervised, ImageNet-1k",
    "dinov2_s_frozen": "self-supervised (DINOv2)",
    "dinov3_s_frozen": "self-supervised (DINOv3)",
    "mae_b_frozen": "masked image modeling",
    "eva02_b_frozen": "masked + contrastive",
    "clip_b32_frozen": "contrastive image-text",
    "siglip_b_frozen": "sigmoid image-text",
    "minilm_l6_frozen": "distilled (MiniLM)",
    "distilbert_base_frozen": "distilled BERT",
    "electra_base_frozen": "replaced-token detection",
    "bert_base_frozen": "masked language modeling",
    "mpnet_base_frozen": "permuted + masked LM",
    "roberta_base_frozen": "masked language modeling",
    "modernbert_base_frozen": "masked language modeling",
    "deberta_v3_base_frozen": "replaced-token detection",
}
_ROSTER_SLUG = {"yearbook": "yearbook", "arxiv": "arxiv", "amazon_reviews_23": "amazon"}


def _rosters(path: Path) -> Path | None:
    """
    Per-dataset model roster (parameter counts, inductive bias, pretraining) as JSON.

    Built from the frozen ``params.parquet`` so the tables on the dataset pages stay in
    sync with the actual model lineup. Returns ``None`` when the parquet is absent.
    """
    from drift_happens.analysis.datasets.locations import PARAMS_PARQUET

    if not PARAMS_PARQUET.exists():
        return None
    frame = pl.read_parquet(PARAMS_PARQUET)
    out: dict[str, dict] = {}
    for dataset in sorted(frame["dataset"].unique().to_list()):
        rows = frame.filter(pl.col("dataset") == dataset)
        image = any(f.startswith("image") for f in rows["trainer_family"].to_list())
        scratch: list[tuple] = []
        frozen: list[tuple] = []
        for row in rows.iter_rows(named=True):
            key, trainable, total = row["trainer"], row["trainable"], row["total"]
            display = get_display_name(key)
            if key.endswith("_frozen"):
                encoder = _ROSTER_ENCODER.get(display, display)
                frozen.append(
                    (
                        total,
                        [
                            encoder,
                            _ROSTER_PRETRAIN.get(key, ""),
                            f"{total:,}",
                            f"{trainable:,}",
                        ],
                    )
                )
                continue
            family, _, size = display.rpartition("-")
            family = _ROSTER_FAMILY.get(family, family)
            bias = _ROSTER_BIAS.get(family, "")
            cells = (
                [family, size, f"{total:,}", bias]
                if image
                else [family, size, f"{trainable:,}", f"{total:,}", bias]
            )
            scratch.append(
                (
                    _ROSTER_FAMILY_ORDER.get(family, 9),
                    _ROSTER_SIZE_ORDER.get(size, 9),
                    cells,
                )
            )
        scratch.sort(key=lambda item: (item[0], item[1]))
        frozen.sort(key=lambda item: item[0])
        scratch_cols = (
            ["Model", "Size", "Parameters", "Inductive bias"]
            if image
            else ["Model", "Size", "Trainable", "Total", "Inductive bias"]
        )
        out[_ROSTER_SLUG.get(dataset, dataset)] = {
            "scratch": {"cols": scratch_cols, "rows": [cells for *_, cells in scratch]},
            "frozen": {
                "cols": [
                    "Encoder",
                    "Pretraining",
                    "Total params",
                    "Trainable (head)",
                ],
                "rows": [cells for _, cells in frozen],
            },
        }
    return _dump(path, out)


def _matrix(
    path: Path,
    spec: DatasetSpec,
    slices: tuple[str, ...],
    values: np.ndarray,
    model: str | None = None,
    raw_range: tuple[float, float] | None = None,
) -> Path:
    payload = {
        "dataset": spec.slug,
        "title": spec.title if model is None else model,
        "metric": spec.metric_label,
        "unit": spec.unit_suffix,
        "higherIsBetter": spec.higher_is_better,
        "valueRange": list(spec.value_range) if spec.value_range else None,
        "rawRange": list(raw_range) if raw_range is not None else None,
        "slices": [slice_label(value, spec) for value in slices],
        "values": _grid(values),
    }
    return _dump(path, payload)


def _bucket_label(spec: DatasetSpec, slices: tuple[str, ...], lo: int, hi: int) -> str:
    """Range label for a coarse bucket spanning slice indices ``lo`` to ``hi``."""
    first = slice_label(slices[lo], spec)
    if lo == hi:
        return first
    return f"{first}–{slice_label(slices[hi], spec)}"


def _replay_dataset(spec: DatasetSpec, mean: derive.DriftMatrix) -> dict:
    """Compact, block-averaged cohort-mean matrix for the hero replay player."""
    buckets, values = derive.coarsen(mean, REPLAY_PERIODS)
    return {
        "slug": spec.slug,
        "title": spec.title,
        "metric": spec.metric_label,
        "unit": spec.unit_suffix,
        "higherIsBetter": spec.higher_is_better,
        "valueRange": list(spec.value_range) if spec.value_range else None,
        "slices": [_bucket_label(spec, mean.slices, lo, hi) for lo, hi in buckets],
        "values": _grid(values),
    }


def _replay(path: Path, datasets: list[dict]) -> Path:
    """Bundle every dataset's coarse replay matrix into one small payload."""
    return _dump(path, {"periods": REPLAY_PERIODS, "datasets": datasets})


def _manifest(
    path: Path,
    dataset: str,
    spec: DatasetSpec,
    lineup: list[derive.LineupEntry],
    deviation_span: float,
) -> Path:
    models = [
        {
            "id": get_display_name(entry.matrix.model),
            "file": (
                figure_name(entry.matrix.model)
                if entry.status is not derive.Status.MISSING
                else None
            ),
            "status": entry.status.value,
        }
        for entry in lineup
    ]
    payload = {
        "dataset": dataset,
        "slug": spec.slug,
        "title": spec.title,
        "metric": spec.metric_label,
        "unit": spec.unit_suffix,
        "higherIsBetter": spec.higher_is_better,
        "valueRange": list(spec.value_range) if spec.value_range else None,
        "deviationSpan": deviation_span,
        "slices": [
            slice_label(value, spec)
            for value in (lineup[0].matrix.slices if lineup else ())
        ],
        "complete": sum(1 for e in lineup if e.status is derive.Status.COMPLETE),
        "planned": len(lineup),
        "models": models,
    }
    return _dump(path, payload)


def _tables(
    path: Path,
    dataset: str,
    spec: DatasetSpec,
    models: list[derive.DriftMatrix],
    frame: pl.DataFrame,
) -> Path:
    scores = {ranking.kind: ranking for ranking in derive.rankings(dataset, models)}
    future = dict(
        zip(
            scores[derive.FUTURE_PERFORMANCE].models,
            scores[derive.FUTURE_PERFORMANCE].score,
            strict=True,
        )
    )
    decay = scores[derive.DECAY]
    robustness = [
        [get_display_name(model), _num(future.get(model), spec), _num(score, spec)]
        for model, score in zip(decay.models, decay.score, strict=True)
    ]

    cutoffs = derive.select_cutoffs(models[0].slices)
    cutoff_tables = []
    for cutoff in cutoffs:
        rows = derive.cutoff_rows(dataset, models, cutoff)
        cutoff_tables.append(
            {
                "cutoff": cutoff,
                "label": slice_label(cutoff, spec),
                "rows": [
                    [
                        rank,
                        get_display_name(row.model),
                        _num(row.in_distribution, spec),
                        _num(row.future, spec),
                        _num(row.decay, spec),
                    ]
                    for rank, row in enumerate(rows, start=1)
                ],
            }
        )

    family = derive.family_rows(frame, dataset, models, cutoffs)
    by_family = {
        "cutoffLabels": [slice_label(cutoff, spec) for cutoff in cutoffs],
        "rows": [
            {
                "family": FAMILY_LABELS.get(row.family, row.family),
                "cells": [
                    [_num(future, spec), _num(decay, spec)]
                    for future, decay in row.cells
                ],
            }
            for row in family
        ],
    }
    payload = {
        "metric": spec.metric_label,
        "unit": spec.unit_suffix,
        "higherIsBetter": spec.higher_is_better,
        "decimals": _decimals(spec),
        "robustness": {"columns": ["Model", "Future", "Decay"], "rows": robustness},
        "cutoffs": cutoff_tables,
        "byFamily": by_family,
    }
    return _dump(path, payload)


def _decimals(spec: DatasetSpec) -> int:
    return int("".join(ch for ch in spec.value_fmt if ch.isdigit()) or "0")


def _num(value: float | None, spec: DatasetSpec) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), _decimals(spec))


def _grid(values: np.ndarray) -> list[list[float | None]]:
    return [
        [None if not np.isfinite(v) else round(float(v), 4) for v in row]
        for row in values
    ]


def _dump(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return path
