# analysis

Paper figures, tables, and website data built from frozen results.

## Layout

- `datasets/` — the metric `schema`, dataset specs, and the artifact paths the pipeline reads and writes.
- `export/` — build the frozen `results.parquet`, `dataset_stats.parquet`, and `params.parquet` (under `artifacts/analysis/`) from the run drift matrices.
- `plots/` — turn the frozen data into figures, tables, and website data.

## Commands

```bash
drift analysis pull         # download finished eval drift matrices from W&B into artifacts/runs
drift analysis export       # build results, dataset statistics, and model parameters
drift analysis figures      # frozen data -> paper/plots_experiments + paper/tables, plus figures.sha256
drift analysis site         # frozen data -> website/data JSON for the client-side site
drift analysis saliency     # render saliency figures for given cutoffs (needs checkpoints)
drift analysis verify       # rebuild figures and tables and check against figures.sha256
```

`drift analysis export` downloads missing public Hugging Face text backbones
when freezing model parameter counts. Pass `--model-params-cache-only` to require
an already-populated local cache.

## Scope

Three datasets: `yearbook` (accuracy), `arxiv` (macro AUC), `amazon_reviews_23` (balanced
MSE, lower is better).
