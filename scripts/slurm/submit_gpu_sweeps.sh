#!/bin/bash
# Submit staged GPU sweep plans through Slurm.

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SCRIPT_DIR/../.." && pwd)"

if (($# > 0)) && [[ ! "$1" =~ ^p[0-9][0-9]_ ]]; then
  CLUSTER_ACCOUNT="$1"
  shift
fi

: "${CLUSTER_ACCOUNT:?set CLUSTER_ACCOUNT or pass the Slurm account as the first arg}"
: "${SCRATCH:?set SCRATCH}"

DRIFT_ENV_FILE="${DRIFT_ENV_FILE:-$SCRATCH/.config/drift-happens-env.sh}"
if [[ ! -f "$DRIFT_ENV_FILE" ]]; then
  echo "Could not find DRIFT_ENV_FILE=$DRIFT_ENV_FILE" >&2
  exit 1
fi

plans=("$@")
if ((${#plans[@]} == 0)); then
  plans=(
    p40_amazon_reviews_23_all_seeds
    p50_arxiv_all_seeds
    p60_yearbook_all_seeds
    p70_imdb_faces_all_seeds
  )
fi

for plan in "${plans[@]}"; do
  plan_file="$_REPO_ROOT/artifacts/experiment_plans/${plan%.yaml}.yaml"
  if [[ ! "$plan" =~ ^p[0-9][0-9]_ || ! -f "$plan_file" ]]; then
    echo "Unknown sweep plan: $plan" >&2
    echo "Pass plan names like p40_amazon_reviews_23_all_seeds." >&2
    echo "Available plans:" >&2
    for available in "$_REPO_ROOT"/artifacts/experiment_plans/p*.yaml; do
      [[ -f "$available" ]] && echo "  ${available##*/}" >&2
    done
    exit 1
  fi
done

gpus_per_node="${DRIFT_GPUS_PER_NODE:-4}"
cpus_per_task="${DRIFT_CPUS_PER_TASK:-128}"

# Derive gpu_indices and sweep_concurrency from gpus_per_node when not set explicitly,
# so a DRIFT_GPUS_PER_NODE override stays consistent with the submitted allocation.
if [[ -n "${DRIFT_GPU_INDICES:-}" ]]; then
  gpu_indices="$DRIFT_GPU_INDICES"
else
  gpu_indices="0"
  for ((gpu_i = 1; gpu_i < gpus_per_node; gpu_i++)); do
    gpu_indices+=",${gpu_i}"
  done
fi
if [[ -n "${DRIFT_SWEEP_CONCURRENCY:-}" ]]; then
  sweep_concurrency="$DRIFT_SWEEP_CONCURRENCY"
else
  sweep_concurrency="$gpus_per_node"
fi

# Prevent an inherited DRIFT_PLAN_FILE from overriding per-plan DRIFT_PLAN_NAME via --export=ALL.
unset DRIFT_PLAN_FILE

export DRIFT_GPU_INDICES="$gpu_indices"
export DRIFT_SWEEP_CONCURRENCY="$sweep_concurrency"

submit_args=(
  sbatch
  -A "$CLUSTER_ACCOUNT"
  --gpus-per-node="$gpus_per_node"
  --cpus-per-task="$cpus_per_task"
  --export=ALL,SCRATCH="$SCRATCH",DRIFT_ENV_FILE="$DRIFT_ENV_FILE"
  "$_REPO_ROOT/scripts/slurm/experiment_sweep_gpu.sbatch"
)

for plan in "${plans[@]}"; do
  export DRIFT_PLAN_NAME="$plan"

  if [[ "${DRIFT_SUBMIT_DRY_RUN:-}" == "1" ]]; then
    printf "DRIFT_PLAN_NAME=%q DRIFT_GPU_INDICES=%q DRIFT_SWEEP_CONCURRENCY=%q " \
      "$DRIFT_PLAN_NAME" "$DRIFT_GPU_INDICES" "$DRIFT_SWEEP_CONCURRENCY"
    printf "%q " "${submit_args[@]}"
    printf "\n"
  else
    "${submit_args[@]}"
  fi
done
