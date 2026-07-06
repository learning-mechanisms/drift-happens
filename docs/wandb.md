# W&B Usage

W&B is optional. It is intended for scalar metrics, run metadata, and small
curated artifacts. Large checkpoints should usually be synchronized with the
rclone/pCloud artifact remote documented in `docs/artifacts.md`.

Set these variables locally or in the cluster environment file:

```bash
export WANDB_PROJECT=drift-happens
export WANDB_ENTITY=drift-happens
export WANDB_MODE=online
export WANDB_TAGS=cluster,slurm
export WANDB_UPLOAD_ARTIFACTS=true
export WANDB_UPLOAD_CHECKPOINTS=false
```

Once W&B is configured, the runtime logs scalar metrics with train/eval slice
context. When `WANDB_UPLOAD_ARTIFACTS=true`, the run artifact also includes
curated files from the local run directory:

- run metadata: `snapshot.json`, `metadata.json`, `run_manifest.json`,
  `config.input.*`
- logs and scalar metric JSONL files
- stage completion JSON files
- result summaries
- `results/drift_matrix.json` and `results/drift_matrix.csv`

Checkpoints are uploaded only when `WANDB_UPLOAD_CHECKPOINTS=true` or the
matching CLI override is used. Keep checkpoint upload disabled for large cluster
sweeps unless W&B is meant to be the checkpoint backend for that run.

On clusters, place W&B directories under scratch storage:

```bash
export WANDB_DIR="${SCRATCH}/wandb"
export WANDB_CACHE_DIR="${SCRATCH}/.cache/wandb"
export WANDB_CONFIG_DIR="${SCRATCH}/.config/wandb"
export WANDB_DATA_DIR="${SCRATCH}/.cache/wandb-data"
```

Use `WANDB_API_KEY` only in a private environment file outside the repository.
