from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from drift_happens.analysis.datasets import schema
from drift_happens.analysis.export import from_matrices, runs
from drift_happens.analysis.plots import (
    build,
    checksums,
    derive,
    names,
    overview,
    site,
    tables,
)
from drift_happens.cli.analysis import app
from drift_happens.experiments.common import BENCHMARK_SEEDS
from drift_happens.experiments.yearbook import YEARBOOK_BENCHMARK_SEEDS


def _rows(dataset, metric, models, slices, *, seeds=None, base=0.9, lag_step=-0.02):
    if seeds is None:
        seeds = YEARBOOK_BENCHMARK_SEEDS if dataset == "yearbook" else BENCHMARK_SEEDS
    rows = []
    for seed in seeds:
        for model in models:
            for i, train in enumerate(slices):
                for j, evaluate in enumerate(slices):
                    rows.append(
                        {
                            "experiment": "e",
                            "dataset": dataset,
                            "dataset_variant": "v",
                            "trainer": model,
                            "trainer_family": "f",
                            "seed": seed,
                            "phase": "eval",
                            "metric": f"eval/{metric}",
                            "value": base + lag_step * max(0, j - i) + 0.01 * seed,
                            "train_slice": train,
                            "eval_slice": evaluate,
                            "step": None,
                            "epoch": None,
                            "config_hash": "h",
                            "snapshot_sha256": "s",
                            "timestamp": "t",
                        }
                    )
    return rows


def _frame(rows):
    return schema.check(pl.DataFrame(rows, schema=schema.SCHEMA))


def test_schema_rejects_wrong_dtype():
    frame = _frame(_rows("yearbook", "accuracy", ["cnn_s"], ["2010", "2011"]))
    with pytest.raises(ValueError):
        schema.check(frame.with_columns(pl.col("seed").cast(pl.Float64)))


def test_all_null_columns_are_cast_to_declared_dtypes():
    frame = pl.DataFrame({column: [None] for column in schema.COLUMNS})
    checked = schema.check(frame)
    assert checked.columns == list(schema.COLUMNS)
    assert pl.Null not in checked.schema.dtypes()


def test_incomplete_seeds_are_skipped_and_reported():
    rows = _rows("yearbook", "accuracy", ["cnn_s"], ["2010", "2011"])
    rows += _rows("yearbook", "accuracy", ["cnn_m"], ["2010", "2011"], seeds=(0,))
    matrices, coverage = derive.per_model_matrices(_frame(rows), "yearbook")
    assert [matrix.model for matrix in matrices] == ["cnn_s"]
    assert coverage.missing == ("cnn_m",)


def test_mean_over_models_blanks_cells_missing_for_a_model():
    rows = _rows("yearbook", "accuracy", ["cnn_s", "cnn_m"], ["2010", "2011", "2012"])
    rows = [
        row
        for row in rows
        if not (row["trainer"] == "cnn_m" and row["eval_slice"] == "2012")
    ]
    matrices, _ = derive.per_model_matrices(_frame(rows), "yearbook")
    mean = derive.mean_over_models(matrices)
    assert np.isnan(mean.mean[0, 2])


def test_decay_is_positive_for_both_metric_directions():
    higher = derive.rankings(
        "yearbook",
        derive.per_model_matrices(
            _frame(_rows("yearbook", "accuracy", ["cnn_s"], ["2010", "2011"])),
            "yearbook",
        )[0],
    )
    lower = derive.rankings(
        "amazon_reviews_23",
        derive.per_model_matrices(
            _frame(
                _rows(
                    "amazon_reviews_23",
                    "balanced_mse",
                    ["ffn_l"],
                    ["28", "29"],
                    base=0.5,
                    lag_step=0.02,
                )
            ),
            "amazon_reviews_23",
        )[0],
    )
    assert next(r for r in higher if r.kind == "decay").score[0] > 0
    assert next(r for r in lower if r.kind == "decay").score[0] > 0


def _drift_matrix(slices, decay=0.02):
    size = len(slices)
    mean = np.array(
        [[1.0 - decay * max(0, j - i) for j in range(size)] for i in range(size)]
    )
    return derive.DriftMatrix("mean", tuple(slices), mean, np.zeros((size, size)))


def test_coarsen_caps_periods_and_covers_every_slice():
    matrix = _drift_matrix([str(year) for year in range(2000, 2030)])  # 30 slices
    buckets, coarse = derive.coarsen(matrix, 12)
    assert coarse.shape == (12, 12)
    assert len(buckets) == 12
    # buckets tile the slice indices contiguously, with no gaps or overlap
    assert buckets[0][0] == 0
    assert buckets[-1][1] == len(matrix.slices) - 1
    for (_, end), (start, _) in zip(buckets, buckets[1:]):
        assert start == end + 1


def test_coarsen_keeps_the_in_distribution_diagonal_warm():
    matrix = _drift_matrix([str(year) for year in range(2000, 2030)])
    _, coarse = derive.coarsen(matrix, 12)
    diagonal = np.diagonal(coarse).mean()
    upper = coarse[np.triu_indices(coarse.shape[0], k=1)].mean()
    assert diagonal > upper  # forward decay survives block-averaging


def test_coarsen_passes_small_matrices_through_one_bucket_per_slice():
    matrix = _drift_matrix(["2018", "2019", "2020"])
    buckets, coarse = derive.coarsen(matrix, 12)
    assert buckets == [(0, 0), (1, 1), (2, 2)]
    np.testing.assert_array_equal(coarse, matrix.mean)


def test_build_site_data_writes_compact_replay(tmp_path):
    slices = [str(year) for year in range(2000, 2018)]  # 18 > REPLAY_PERIODS
    frame = _frame(_rows("yearbook", "accuracy", ["cnn_s", "cnn_m"], slices))
    paths = site.build_site_data(frame, tmp_path)
    replay_path = tmp_path / "data" / "replay.json"
    assert replay_path in paths
    payload = json.loads(replay_path.read_text())
    assert payload["periods"] == site.REPLAY_PERIODS
    yearbook = next(d for d in payload["datasets"] if d["slug"] == "yearbook")
    assert yearbook["metric"] == "Accuracy"
    assert len(yearbook["slices"]) == site.REPLAY_PERIODS
    assert all(len(row) == site.REPLAY_PERIODS for row in yearbook["values"])
    assert "–" in yearbook["slices"][0]  # multi-year buckets carry a range label


def test_repeated_build_matches(tmp_path):
    frame = _frame(
        _rows("arxiv", "auc_macro", ["bigru_s", "ffn_l"], ["2018", "2019", "2020"])
    )
    first = build.build(
        tmp_path / "a" / "figures",
        tmp_path / "a" / "tables",
        tmp_path / "a" / "pages",
        frame=frame,
    )
    second = build.build(
        tmp_path / "b" / "figures",
        tmp_path / "b" / "tables",
        tmp_path / "b" / "pages",
        frame=frame,
    )
    assert checksums.manifest(first.outputs, tmp_path / "a") == checksums.manifest(
        second.outputs, tmp_path / "b"
    )
    assert first.outputs


def test_checksum_detects_a_changed_figure(tmp_path):
    frame = _frame(_rows("yearbook", "accuracy", ["cnn_s", "cnn_m"], ["2010", "2011"]))
    report = build.build(
        tmp_path / "figures", tmp_path / "tables", tmp_path / "pages", frame=frame
    )
    manifest = tmp_path / "out.sha256"
    checksums.write(report.outputs, tmp_path, manifest)
    assert checksums.verify(report.outputs, tmp_path, manifest) == []
    victim = next(path for path in report.outputs if path.suffix == ".pdf")
    victim.write_bytes(victim.read_bytes() + b"x")
    changed = checksums.verify(report.outputs, tmp_path, manifest)
    assert victim.relative_to(tmp_path).as_posix() in changed


def test_cutoff_and_family_tables(tmp_path):
    frame = _frame(
        _rows(
            "yearbook",
            "accuracy",
            ["cnn_s", "cnn_m", "mlp_s"],
            [str(y) for y in range(2008, 2013)],
        )
    )
    models, _ = derive.per_model_matrices(frame, "yearbook")
    cutoffs = derive.select_cutoffs(models[0].slices, count=2)
    rows = derive.cutoff_rows("yearbook", models, cutoffs[0], top_n=2)
    assert 0 < len(rows) <= 2
    cutoff_tex = tables.cutoff_table(
        "yearbook", cutoffs[0], rows, tmp_path / "c.tex"
    ).read_text()
    assert "Rank & Model" in cutoff_tex
    family = derive.family_rows(frame, "yearbook", models, cutoffs)
    family_tex = tables.family_table(
        "yearbook", cutoffs, family, tmp_path / "f.tex"
    ).read_text()
    assert r"\multicolumn{2}{c}" in family_tex


def _stats_rows():
    rows = []
    for year in range(2008, 2012):
        for gender in ("M", "F"):
            rows.append(
                {
                    "dataset": "yearbook",
                    "slice_kind": "year",
                    "slice": year,
                    "group_kind": "gender",
                    "group": gender,
                    "count": 100 + year,
                }
            )
    for subject, count in [("cs", 50), ("math", 30)]:
        rows.append(
            {
                "dataset": "arxiv",
                "slice_kind": "year",
                "slice": None,
                "group_kind": "subject",
                "group": subject,
                "count": count,
            }
        )
    return rows


def test_lineup_flags_status_and_appendix_shows_only_complete(tmp_path):
    from drift_happens.analysis.datasets import DATASETS
    from drift_happens.analysis.plots import appendix

    rows = _rows("yearbook", "accuracy", ["cnn_s"], ["2010", "2011"])
    rows += _rows("yearbook", "accuracy", ["cnn_m"], ["2010", "2011"], seeds=(0,))
    lineup = derive.lineup_matrices(
        _frame(rows), "yearbook", ["cnn_s", "cnn_m", "mlp_l"]
    )
    status = {item.matrix.model: item.status for item in lineup}
    assert status == {
        "cnn_s": derive.Status.COMPLETE,
        "cnn_m": derive.Status.PARTIAL,
        "mlp_l": derive.Status.MISSING,
    }
    text = appendix.appendix_page(
        DATASETS["yearbook"],
        lineup,
        ["2010"],
        {"cnn_s": "image-cnn", "cnn_m": "image-cnn"},
        False,
        tmp_path / "app.tex",
    ).read_text()
    assert r"\textcolor{red}" not in text
    assert "partial run" not in text and "no completed run" not in text
    assert r"\subsection{CNN}" in text
    assert r"\caption{CNN-S}" in text
    assert "CNN-M" not in text and "MLP-L" not in text
    assert r"\ref{subsec:drift_summary}" in text
    assert r"\input{tables/robustness_yearbook.tex}" in text
    assert r"\input{tables/yearbook_roster.tex}" not in text


def test_overview_renders_and_is_deterministic(tmp_path):
    stats = schema.check_stats(pl.DataFrame(_stats_rows()))
    first = overview.build_overview(stats, tmp_path / "a")
    second = overview.build_overview(stats, tmp_path / "b")
    assert first
    assert checksums.manifest(first, tmp_path / "a") == checksums.manifest(
        second, tmp_path / "b"
    )


def test_stats_schema_rejects_missing_column():
    with pytest.raises(ValueError):
        schema.check_stats(pl.DataFrame({"dataset": ["yearbook"]}))


def test_trainer_family_groups_text_by_architecture():
    assert names.trainer_family("arxiv", "ffn_s") == "text-ffn"
    assert names.trainer_family("arxiv", "textcnn_l") == "text-textcnn"
    assert names.trainer_family("arxiv", "tx_m") == "text-tx"
    for recurrent in ("bigru_s", "bilstm_m", "bilstm_attn_l"):
        assert names.trainer_family("arxiv", recurrent) == "text-rnn"
    assert names.trainer_family("arxiv", "bert_base_frozen") == "text-frozen-head"
    assert names.trainer_family("amazon_reviews_23", "bigru_s") == "text-rnn-regression"
    assert names.trainer_family("yearbook", "cnn_s") == "image-cnn"
    assert {names.FAMILY_LABELS[k] for k in ("text-rnn", "text-tx")} == {
        "Recurrent",
        "Transformer",
    }


def test_roster_table_splits_scratch_and_frozen(tmp_path):
    rows = schema.check_params(
        pl.DataFrame(
            [
                {
                    "dataset": "arxiv",
                    "trainer": "bigru_s",
                    "trainer_family": "text-rnn",
                    "trainable": 4_100_000,
                    "total": 4_100_000,
                },
                {
                    "dataset": "arxiv",
                    "trainer": "bert_base_frozen",
                    "trainer_family": "text-frozen-head",
                    "trainable": 420_000,
                    "total": 110_000_000,
                },
            ]
        )
    )
    text = tables.roster_table("arxiv", rows, tmp_path / "r.tex").read_text()
    assert "models trained from scratch" in text
    assert "frozen pretrained encoders" in text
    assert "Model & Family & Parameters" in text
    assert "Model & Family & Trainable & Total" in text
    assert "4.1M" in text and "420k" in text and "110.0M" in text


def test_params_builds_a_text_model():
    from drift_happens.analysis.export.params import _text_model
    from drift_happens.model.parameters import count_parameters

    model = _text_model({"architecture": "bigru_s", "input_dim": 768, "output_dim": 5})
    assert count_parameters(model).total > 0


def test_backbone_params_uses_cache_before_downloading(monkeypatch):
    from drift_happens.analysis.export.params import _backbone_params

    class Parameter:
        def __init__(self, count: int) -> None:
            self.count = count

        def numel(self) -> int:
            return self.count

    class Model:
        def parameters(self):
            return [Parameter(2), Parameter(5)]

    calls = []

    def from_pretrained(producer: str, *, local_files_only: bool):
        calls.append((producer, local_files_only))
        if local_files_only:
            raise OSError("not cached")
        return Model()

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", from_pretrained)

    assert _backbone_params("example/model") == 7
    assert calls == [("example/model", True), ("example/model", False)]


def test_backbone_params_cache_only_does_not_retry_online(monkeypatch):
    from drift_happens.analysis.export.params import _backbone_params

    calls = []

    def from_pretrained(producer: str, *, local_files_only: bool):
        calls.append((producer, local_files_only))
        raise OSError("not cached")

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", from_pretrained)

    with pytest.raises(OSError):
        _backbone_params("example/model", local_files_only=True)
    assert calls == [("example/model", True)]


def test_yearbook_params_disable_pretrained_transfer_backbones(monkeypatch):
    from drift_happens.analysis.export.params import _yearbook_rows
    from drift_happens.model.dataset.image.transfer_learning.base import (
        TransferLearningConfig,
    )

    class Parameter:
        requires_grad = True

        def numel(self) -> int:
            return 3

    class Model:
        def parameters(self):
            return [Parameter()]

    class Preset:
        group = "conference"

        def build(self):
            trainer = SimpleNamespace(key="clip_b32_frozen")
            return SimpleNamespace(trainer=trainer)

    class Module:
        @staticmethod
        def presets():
            return (Preset(),)

    seen = []

    def trainer_configs():
        return {
            "clip_b32_frozen": SimpleNamespace(
                architecture_specific_config=TransferLearningConfig(pretrained=True)
            )
        }

    def image_model_factory(config):
        seen.append(config)
        return Model()

    monkeypatch.setattr(
        "drift_happens.pipeline.yearbook.trainers.yearbook_conference_trainer_configs",
        trainer_configs,
    )
    monkeypatch.setattr(
        "drift_happens.pipeline.image.trainers.image_model_factory",
        image_model_factory,
    )

    rows = _yearbook_rows(Module)

    assert seen[0].pretrained is False
    assert rows[0]["trainer"] == "clip_b32_frozen"
    assert rows[0]["total"] == 3


@pytest.mark.parametrize(
    ("args", "expected_cache_only"),
    [
        (["export"], False),
        (["export", "--model-params-cache-only"], True),
    ],
)
def test_analysis_export_forwards_model_param_loading_policy(
    monkeypatch, args, expected_cache_only
):
    from drift_happens.analysis.export import __main__ as export_module
    from drift_happens.analysis.export import dataset_stats, params

    calls = []

    monkeypatch.setattr(export_module, "export", lambda: {"runs": [], "missing": []})
    monkeypatch.setattr(dataset_stats, "freeze_dataset_stats", lambda: None)

    def freeze_params(*, local_files_only: bool) -> None:
        calls.append(local_files_only)

    monkeypatch.setattr(params, "freeze_params", freeze_params)

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0
    assert calls == [expected_cache_only]


def test_render_does_not_import_torch():
    code = (
        "import drift_happens.analysis.plots.build, drift_happens.analysis.plots.site; "
        "import sys; assert 'torch' not in sys.modules"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_build_results_from_matrices_and_lock(tmp_path):
    run_dir = tmp_path / "yearbook-conference__cnn_s__seed=0__eval"
    (run_dir / "results").mkdir(parents=True)
    matrix = {
        "2010": {"2010": {"accuracy": 0.91}, "2011": {"accuracy": 0.88}},
        "2011": {"2010": {"accuracy": 0.87}, "2011": {"accuracy": 0.90}},
    }
    (run_dir / "results" / "drift_matrix.json").write_text(json.dumps(matrix))

    output = from_matrices.build_results_from_matrices(
        runs_root=tmp_path, output=tmp_path / "results.parquet"
    )
    frame = pl.read_parquet(output)
    schema.check(frame)
    assert frame.height == 4
    assert set(frame["metric"].unique().to_list()) == {"accuracy"}
    assert frame["trainer_family"].unique().to_list() == ["image-cnn"]
    assert frame["dataset"].unique().to_list() == ["yearbook"]

    lock = runs.lock(frame)
    pinned = {(run["trainer"], run["seed"]) for run in lock["runs"]}
    assert pinned == {("cnn_s", 0)}
    assert {"dataset": "yearbook", "trainer": "cnn_s", "seed": 0} not in lock["missing"]


def test_build_results_from_matrices_rejects_unknown_family(tmp_path):
    run_dir = tmp_path / "yearbook-conference__bogus__seed=0__eval"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "results" / "drift_matrix.json").write_text(
        json.dumps({"2010": {"2010": {"accuracy": 0.9}}})
    )
    with pytest.raises(ValueError, match="no family label"):
        from_matrices.build_results_from_matrices(
            runs_root=tmp_path, output=tmp_path / "results.parquet"
        )


def test_build_results_from_matrices_needs_a_matrix(tmp_path):
    with pytest.raises(FileNotFoundError, match="no drift_matrix.json"):
        from_matrices.build_results_from_matrices(
            runs_root=tmp_path, output=tmp_path / "results.parquet"
        )


def test_deviation_extent_handles_no_matrices():
    assert derive.deviation_extent([]) == 1.0


def test_build_results_from_matrices_skips_malformed_json(tmp_path):
    good = tmp_path / "yearbook-conference__cnn_s__seed=0__eval"
    (good / "results").mkdir(parents=True)
    (good / "results" / "drift_matrix.json").write_text(
        json.dumps({"2010": {"2010": {"accuracy": 0.9}}})
    )
    bad = tmp_path / "yearbook-conference__mlp_s__seed=0__eval"
    (bad / "results").mkdir(parents=True)
    (bad / "results" / "drift_matrix.json").write_text("{not valid json")
    output = from_matrices.build_results_from_matrices(
        runs_root=tmp_path, output=tmp_path / "results.parquet"
    )
    frame = pl.read_parquet(output)
    assert frame["trainer"].unique().to_list() == ["cnn_s"]


def test_expected_cells_cover_three_conference_datasets():
    cells = runs.expected_cells()
    datasets = {dataset for dataset, _, _ in cells}
    assert datasets == {"yearbook", "arxiv", "amazon_reviews_23"}
    seeds_by_dataset = {
        dataset: {seed for cell_dataset, _, seed in cells if cell_dataset == dataset}
        for dataset in datasets
    }
    assert seeds_by_dataset == {
        "amazon_reviews_23": set(BENCHMARK_SEEDS),
        "arxiv": set(BENCHMARK_SEEDS),
        "yearbook": set(YEARBOOK_BENCHMARK_SEEDS),
    }


def test_tex_escapes_latex_specials_and_passes_clean_names():
    from drift_happens.analysis.plots.latex import tex

    assert tex("acc_%&#$") == r"acc\_\%\&\#\$"
    assert tex("x\\y") == r"x\textbackslash{}y"
    assert tex("~^{}") == r"\textasciitilde{}\textasciicircum{}\{\}"
    assert tex("BiGRU-S") == "BiGRU-S"


def test_robustness_table_uses_display_names_and_units(tmp_path):
    frame = _frame(_rows("arxiv", "auc_macro", ["bigru_s", "ffn_l"], ["2018", "2019"]))
    matrices, _ = derive.per_model_matrices(frame, "arxiv")
    scores = derive.rankings("arxiv", matrices)
    text = tables.robustness_table("arxiv", scores, tmp_path / "r.tex").read_text()
    assert r"\caption{Temporal robustness on arXiv.}" in text
    assert r"Future (\%)" in text
    assert "BiGRU-S" in text


def test_verify_without_manifest_exits_cleanly(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "drift_happens.cli.analysis.FIGURES_MANIFEST", tmp_path / "missing.sha256"
    )
    result = CliRunner().invoke(app, ["verify"])
    assert result.exit_code == 1
    assert "not found" in result.output
