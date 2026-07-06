# Runtime Results Plotting

The reproducible plotting entry point is:

```bash
pixi run plots-results
```

It scans `artifacts/runs/` for staged runtime outputs with
`results/drift_matrix.json`, writes one plot per run, writes raw matrix CSV files, and
creates `artifacts/plots/runtime/index.md`. Plots are PDF-only by default.

Paper-scope plots use the same runtime input contract but write into a new
unreferenced directory under `paper/generated/`:

```bash
pixi run paper-result-plots
pixi run paper-dataset-plots
```

`paper-result-plots` writes vector PDF heatmaps to
`paper/generated/results_v2/` and does not update LaTeX `\includegraphics`
references. `paper-dataset-plots` writes deterministic dataset overview PDFs to
`paper/generated/dataset_overview/` after the corresponding datasets have been
prepared.

## Common Commands

```bash
# Preview what would be plotted without writing files.
pixi run plots-results-dry-run

# Plot one metric across all local runs.
pixi run drift eval plots --metric accuracy

# Filter to a dataset or trainer.
pixi run drift eval plots --dataset yearbook --trainer cnn_s

# Write PDF output to a custom directory.
pixi run drift eval plots --out-dir artifacts/plots/yearbook

# Optional ad hoc formats can still be requested explicitly.
pixi run drift eval plots --format svg --out-dir artifacts/plots/yearbook-svg

# Plot a specific run directory or a copied artifact bundle.
pixi run drift eval plots artifacts/runs/yearbook/cnn_s/my-run/seed=0/run-leaf
```

Paper-matrix scripts for the
`artifacts/experiments/<dataset>/<scope_variant>/` layout (for example
`artifacts/experiments/yearbook/faces_32x32_cumulative_v2/`) are also registered as
Pixi tasks:

```bash
pixi run plots-yearbook-matrices
pixi run plots-arxiv-matrices
pixi run plots-amazon-reviews-23-matrices
pixi run plots-imdb-faces-matrices
```

Those tasks generate individual raw matrices plus mean/deviation matrices when the
corresponding scope-variant experiment directories are present. They use the same
paper plotting style as `drift eval plots`. Output is written under
`plots/<dataset>/drift_matrices/` (for example `plots/yearbook/drift_matrices/`),
which is also ignored by git.

## Input Contract

The plotting command is intentionally built on the current runtime contract rather than
the older notebook-specific `artifacts/experiments/` layout. Each plotted run needs:

- `snapshot.json` for dataset, trainer, seed, tags, and primary metric fallback.
- `metadata.json` or `run_manifest.json` for the source identity when available.
- `results/summary.json` for the primary metric (takes precedence over snapshot.json).
- `results/drift_matrix.json` with the shape
  `{train_slice: {eval_slice: {metric_name: value}}}`.

Metric names can be requested either exactly or without their runtime phase prefix. For
example, `--metric accuracy` matches both `accuracy` and `eval/accuracy`.

## Outputs

For each run, outputs are written under:

```text
artifacts/plots/runtime/<dataset>/<trainer>/<source_identity>/seed=<n>/<metric>/
```

The directory contains:

- `matrix.csv`: raw scalar values from `drift_matrix.json`.
- `metadata.json`: run and metric provenance for the generated plot.
- `drift_matrix.pdf`: rendered heatmap.

When multiple seeds exist for the same dataset, trainer, experiment, source identity,
and metric, the command also writes seed aggregate matrices under `aggregate/<metric>/`.
Those aggregate outputs include `mean_matrix.csv`, `std_matrix.csv`,
`mean_drift_matrix.pdf`, and `std_drift_matrix.pdf`.

Score-like metrics such as accuracy, AUC, precision, recall, and F1 are rendered as
percentages when all values are in `[0, 1]`. Error-like metrics such as loss, MSE, RMSE,
and MAE use the lower-is-better color scale.

Generated plots use a LaTeX-like serif/math font stack without requiring a TeX install.
`drift eval plots` outputs live under `artifacts/plots/`, which is ignored by git.
Curate only the final figures needed for the paper or website.
