# Architecture

`drift_happens/runtime/` is the public execution entry point for local runs. It
coordinates staged train and eval work, run identity, resume behavior, metrics,
and artifact summaries.

## Runtime Flow

1. `drift experiment run`, `train`, or `eval` loads an `ExperimentConfig`.
2. The local runtime resolves a dataset adapter from
   `drift_happens.runtime.adapters`.
3. The adapter runs train and eval stages under `artifacts/runs/.../stages/`.
4. Stage completions and metric logs make resume/status checks deterministic.
5. Result summaries are written under `artifacts/runs/.../results/`.

## Adapter Contract

Dataset adapters expose:

- support detection for an `ExperimentConfig`
- expected train unit enumeration
- expected eval unit enumeration
- train execution
- eval execution

The synthetic adapter is the smallest reference implementation. Dataset pipeline
adapters bridge older Yearbook, IMDB Faces, arXiv, and Amazon Reviews pipelines
into the staged runtime while preserving existing experiment behavior.

## Artifact Layout

The canonical runtime layout is `artifacts/runs/...`. Legacy
`artifacts/experiments/...` output is treated as generated research output and is
not part of the public source boundary.

## Packaging

The project is Pixi-first. Editable package installation is supported inside the
Pixi environment through `pixi run postinstall`; standalone pip dependency
resolution is not currently the supported installation path.
