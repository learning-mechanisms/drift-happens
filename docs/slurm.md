# Slurm Runbook

This repository includes Slurm scripts for cluster setup checks and dataset
preparation under `scripts/slurm/`.

The scripts do not contain secrets, W&B project names, Kaggle credentials, or
Hugging Face tokens. They also omit the Slurm account because `#SBATCH` lines
are parsed before normal shell variable expansion. Keep private values in an
environment file and pass the cluster account at submit time.

## Environment File

Create the environment file outside the repository:

```bash
mkdir -p "$SCRATCH/.config"
chmod 700 "$SCRATCH/.config"
$EDITOR "$SCRATCH/.config/drift-happens-env.sh"
chmod 600 "$SCRATCH/.config/drift-happens-env.sh"
```

The Slurm scripts use `DRIFT_ENV_FILE` when it is set. Otherwise they look for
`$SCRATCH/.config/drift-happens-env.sh`,
`$PROJECT_SCRATCH/.config/drift-happens-env.sh`, and then
`$HOME/.config/drift-happens-env.sh`.

Use this shape:

```bash
export DRIFT_REPO_DIR="${SCRATCH}/git/drift-happens"
export DRIFT_DATA_DIR="${SCRATCH}/drift-happens/data"
export DRIFT_ARTIFACTS_DIR="${SCRATCH}/drift-happens/artifacts"
export DRIFT_PLOTS_DIR="${SCRATCH}/drift-happens/plots"

export XDG_CACHE_HOME="${SCRATCH}/.cache"
export MPLCONFIGDIR="${SCRATCH}/.cache/matplotlib"
export NUMBA_CACHE_DIR="${SCRATCH}/.cache/numba"
export TORCH_HOME="${SCRATCH}/.cache/torch"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_HUB_CACHE="${SCRATCH}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${SCRATCH}/.cache/huggingface/datasets"
export TOKENIZERS_PARALLELISM=false

export WANDB_DIR="${SCRATCH}/wandb"
export WANDB_CACHE_DIR="${SCRATCH}/.cache/wandb"
export WANDB_CONFIG_DIR="${SCRATCH}/.config/wandb"
export WANDB_DATA_DIR="${SCRATCH}/.cache/wandb-data"
export WANDB_PROJECT=drift-happens
export WANDB_ENTITY=drift-happens
export WANDB_MODE=online
export WANDB_TAGS=cluster,slurm
export WANDB_UPLOAD_ARTIFACTS=false
export WANDB_UPLOAD_CHECKPOINTS=false

export RCLONE_CONFIG="${SCRATCH}/.config/rclone/rclone.conf"

export KAGGLE_CONFIG_DIR="${SCRATCH}/.config/kaggle"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
```

Add private credentials only when needed:

```bash
export WANDB_API_KEY=...
export HUGGINGFACE_TOKEN=...
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
```

If artifacts uploads are enabled with `WANDB_UPLOAD_ARTIFACTS=true`, W&B uploads
local run metadata and metrics as run artifacts. Keeping this disabled is the
lowest-friction default for smoke and large sweeps.

`WANDB_UPLOAD_ARTIFACTS=true` is enough to upload curated metadata, logs,
metrics, and `results/drift_matrix.*` files after a run stage closes. Model
checkpoints are still skipped unless `WANDB_UPLOAD_CHECKPOINTS=true` is set.
For full cluster sweeps, prefer W&B for metrics and small matrix artifacts, and
use the pCloud/rclone artifact remote for large checkpoint archives.

## One-Time Setup

Run these commands once after cloning the repository:

```bash
source "$SCRATCH/.config/drift-happens-env.sh"
cd "$DRIFT_REPO_DIR"

mkdir -p logs \
  "$DRIFT_DATA_DIR" "$DRIFT_ARTIFACTS_DIR" "$DRIFT_PLOTS_DIR" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" \
  "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" \
  "$TORCH_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$KAGGLE_CONFIG_DIR" "$(dirname "$RCLONE_CONFIG")"

pixi install
pixi run postinstall
pixi run materialize
pixi run materialize-check
pixi run experiment-plans
pixi run experiment-plans-check
```

If this cluster cannot open a browser from compute nodes, configure rclone on a
login node or locally and copy the resulting `rclone.conf` to `RCLONE_CONFIG`.
Then initialize the project artifact profile:

```bash
pixi run drift artifacts remote setup --skip-rclone-config
pixi run artifacts-remote-status
```

For GPU jobs, install the CUDA-enabled Pixi environment before submitting. On
login nodes where Pixi cannot see a CUDA driver, mock the CUDA virtual package:

```bash
CONDA_OVERRIDE_CUDA=12.0 pixi install -e gpu
pixi run -e gpu postinstall
```

## GPU Smoke Test

Submit the GPU smoke test with the Slurm account at submit time:

```bash
source "$SCRATCH/.config/drift-happens-env.sh"
cd "$DRIFT_REPO_DIR"
export CLUSTER_ACCOUNT=<account>
sbatch -A "$CLUSTER_ACCOUNT" scripts/slurm/smoke_gpu.sbatch
tail -F run.log
```

If the job cannot find the environment file, pass it explicitly:

```bash
sbatch -A "$CLUSTER_ACCOUNT" \
  --export=ALL,DRIFT_ENV_FILE="$SCRATCH/.config/drift-happens-env.sh" \
  scripts/slurm/smoke_gpu.sbatch
```

The script materializes a CUDA smoke plan under
`$DRIFT_ARTIFACTS_DIR/experiment_plans/cuda-debug/` and runs
`p00_smoke_seed0.yaml` from there. It does not modify the tracked curated plans
under `artifacts/experiment_plans/`.

Validate completion:

```bash
pixi run drift experiment stages status \
  configs/snapshots/presets/smoke/synthetic-classification-cpu.json \
  --seed 0

pixi run drift experiment seeds status \
  configs/snapshots/presets/smoke/synthetic-classification-cpu.json
```

Both train and eval should report `ok`.

## GPU Experiment Sweeps

Run the full experiment campaign as two disjoint staged sweeps:

```bash
export DRIFT_PLAN_NAME=p80_seed0_all_presets
export DRIFT_GPU_INDICES=0
export DRIFT_SWEEP_CONCURRENCY=1
# Resume reuses completed work by default; set to 0 to force fresh re-runs.
# export DRIFT_ENABLE_AUTO_RESUME=0
# Optional: set to 1 to continue an unfinished slice from its epoch checkpoint.
# export DRIFT_RESUME_CHECKPOINTS=1
sbatch -A "$CLUSTER_ACCOUNT" \
  --export=ALL,SCRATCH="$SCRATCH",DRIFT_ENV_FILE="$SCRATCH/.config/drift-happens-env.sh" \
  scripts/slurm/experiment_sweep_gpu.sbatch

export DRIFT_PLAN_NAME=p90_remaining_seeds_all_presets
export DRIFT_GPU_INDICES=0,1,2,3
export DRIFT_SWEEP_CONCURRENCY=4
sbatch -A "$CLUSTER_ACCOUNT" \
  --gpus-per-node=4 \
  --cpus-per-task=128 \
  --export=ALL,SCRATCH="$SCRATCH",DRIFT_ENV_FILE="$SCRATCH/.config/drift-happens-env.sh" \
  scripts/slurm/experiment_sweep_gpu.sbatch
```

The second job requests four GPUs and materializes four CUDA sweep slots. The
sweep runner dispatches up to four child runs at a time and sets
`CUDA_VISIBLE_DEVICES` for each child process.

For interactive nodes, materialize host-local plans with only the seed(s) assigned
to that host:

```bash
export HOST_PLAN_DIR="$DRIFT_ARTIFACTS_DIR/experiment_plans/interactive-$(hostname)"

pixi run -e gpu drift experiment plans materialize --write \
  --out-dir "$HOST_PLAN_DIR" \
  --device cuda \
  --gpu-indices 0,1,2,3 \
  --concurrency 4 \
  --seeds 1,3

pixi run -e gpu drift experiment sweep \
  "$HOST_PLAN_DIR/p60_yearbook_all_seeds.yaml" \
  --skip-source wandb
```

The generated sweep files are only dispatch manifests. They do not change run
identity, so seed-filtered plans generated on different hosts can still be merged
under the same `artifacts/runs/` tree or through W&B artifacts.

To submit one all-seed GPU sweep per main dataset:

```bash
export CLUSTER_ACCOUNT=<account>
export SCRATCH=/pscratch/sd/<initial>/<username>
scripts/slurm/submit_gpu_sweeps.sh
```

Preview the submissions without queuing jobs:

```bash
DRIFT_SUBMIT_DRY_RUN=1 scripts/slurm/submit_gpu_sweeps.sh "$CLUSTER_ACCOUNT"
```

The default plan sequence is:

```text
p40_amazon_reviews_23_all_seeds  # 62 jobs
p50_arxiv_all_seeds              # 65 jobs
p60_yearbook_all_seeds           # 110 jobs
p70_imdb_faces_all_seeds         # 63 jobs
```

Each submitted job requests four GPUs by default and runs up to four child
experiments concurrently. The sweeps use `skip_completed=true`, so completed
local runs are skipped when a later broader plan includes them.

To submit only selected stages, pass their names after the account:

```bash
scripts/slurm/submit_gpu_sweeps.sh "$CLUSTER_ACCOUNT" \
  p40_amazon_reviews_23_all_seeds \
  p50_arxiv_all_seeds
```

The original broad coverage plans remain available:

```text
p80_seed0_all_presets
p90_remaining_seeds_all_presets
p99_everything_all_seeds
```

## Dataset Setup

Submit dataset preparation as a CPU batch job:

```bash
source "$SCRATCH/.config/drift-happens-env.sh"
cd "$DRIFT_REPO_DIR"
export CLUSTER_ACCOUNT=<account>
sbatch -A "$CLUSTER_ACCOUNT" scripts/slurm/datasets.sbatch
tail -F datasets.log
```

The same explicit environment-file override works for dataset setup:

```bash
sbatch -A "$CLUSTER_ACCOUNT" \
  --export=ALL,DRIFT_ENV_FILE="$SCRATCH/.config/drift-happens-env.sh" \
  scripts/slurm/datasets.sbatch
```

The dataset setup command passes `--yes` to every step, so it overwrites existing
files without prompting. Run this job against a clean `DRIFT_DATA_DIR`, or clear
the specific dataset directory before rerunning.
