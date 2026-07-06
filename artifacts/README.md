# artifacts/

This directory holds generated outputs from experiments, sweeps, plots, and reports.
Large runtime outputs should stay out of git unless they are deliberately curated for
the paper, website, or regression fixtures.

## Canonical Runtime Layout

Experiment runners write one stable directory per resolved config and seed:

```text
artifacts/
+-- README.md
+-- runs/
|   +-- <dataset_name>/
|       +-- <trainer_key>/
|           +-- <experiment_name>/
|               +-- seed=<n>/
|                   +-- <source_identity>__cfg=<hash>/
|                       +-- snapshot.json
|                       +-- metadata.json
|                       +-- run_manifest.json
|                       +-- config.input.yaml
|                       +-- stages/
|                       |   +-- train/
|                       |   |   +-- metadata.json
|                       |   |   +-- completion.json
|                       |   +-- eval/
|                       |       +-- metadata.json
|                       |       +-- completion.json
|                       +-- attempts/
|                       |   +-- train/
|                       |   +-- eval/
|                       +-- logs/
|                       |   +-- train.console.log
|                       |   +-- eval.console.log
|                       |   +-- events.jsonl
|                       +-- metrics/
|                       |   +-- train.jsonl
|                       |   +-- eval.jsonl
|                       +-- results/
|                           +-- summary.json
|                           +-- drift_matrix.json
+-- sweeps/
|   +-- <UTC-timestamp>__<sweep_name>/
|       +-- manifest.json
|       +-- results.json
|       +-- logs/
|           +-- console.log
|           +-- events.jsonl
|           +-- <job_label>__seed<n>.log
+-- experiment_plans/
|   +-- <plan_name>.yaml
+-- plots/
|   +-- runtime/
|       +-- index.md
|       +-- <dataset>/<trainer>/<source_identity>/seed=<n>/<metric>/
|           +-- matrix.csv
|           +-- metadata.json
|           +-- drift_matrix.pdf
+-- reports/
    +-- <YYYY-MM-DD>/
        +-- report.md
```

## File Contracts

`snapshot.json` is the deterministic resolved `ExperimentConfig`. It should be enough
to replay the run's intended configuration.

`metadata.json` stores volatile execution facts: git commit, dirty status, truncated
diff, pixi lock hash, host information, device information, timestamps, wall time,
exit status, and the last completed iteration. Run-level metadata is `ok` only when
both `stages/train/completion.json` and `stages/eval/completion.json` are `ok` for the
same source identity, config hash, snapshot hash, and seed.

`logs/events.jsonl` is for structured application events. Stage console logs mirror
human-readable terminal logs. `metrics/` is for iteration-level numeric outputs used by
analysis code. Partial files, checkpoints, and directory existence never count as
success without explicit completion markers.

## Resume Rules

Resume is enabled by default for command-line and sweep runs: a plain re-run reuses
completed work units from a started run instead of recomputing them. Pass `--no-resume`
(or set `DRIFT_ENABLE_AUTO_RESUME=0` when no flag is given) to force fresh re-runs that
clear the owned stage outputs and recompute from scratch.

Mid-training epoch checkpoints are a separate, opt-in mechanism: by default an
*unfinished* slice retrains from epoch 0 rather than continuing from its last epoch
checkpoint. Pass `--resume-checkpoints` (or set `DRIFT_RESUME_CHECKPOINTS=1`) to continue
an interrupted slice. This is independent of completed-unit reuse, which always applies.

`drift experiment train CONFIG --seed N` owns only the train stage. A completed train
stage is skipped by default. `--no-resume` clears train outputs and also invalidates
eval outputs for the same run because eval depends on the trained models.

`drift experiment eval CONFIG --seed N` owns only the eval stage. It refuses to run
until the matching train stage is complete. `--no-resume` clears eval outputs while
preserving train outputs.

`drift experiment run CONFIG --seed N` is a convenience command that launches train and
eval as separate CLI processes by default. Local artifacts remain canonical; W&B mirrors
metrics and completion fields when configured.

Use `drift experiment seeds status CONFIG`, `drift experiment seeds summarize CONFIG`,
`drift experiment plans materialize --check`, `drift experiment sweep PLAN.yaml`, and
`drift artifacts gc --dry-run` for the normal multi-seed workflow.

Use `drift eval plots` or `pixi run plots-results` to render reproducible drift-matrix
figures from `results/drift_matrix.json` files. See `docs/results-plotting.md` for the
plotting contract and outputs.

## Public Repository Boundary

Generated outputs under `artifacts/experiments/`, `artifacts/plots/`,
`artifacts/robustness_results/`, `artifacts/runs/`, and `artifacts/sweeps/` are ignored
by default. Publish large reproducibility bundles through release assets, Zenodo, or a
dedicated artifact repository, then link them from `docs/artifacts.md`.
