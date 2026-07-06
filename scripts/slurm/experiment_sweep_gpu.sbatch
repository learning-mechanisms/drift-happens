#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=drift-gpu-sweep
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -euo pipefail

env_candidates=()
if [[ -z "${DRIFT_ENV_FILE:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    env_candidates+=("$SCRATCH/.config/drift-happens-env.sh")
  fi
  if [[ -n "${PROJECT_SCRATCH:-}" ]]; then
    env_candidates+=("$PROJECT_SCRATCH/.config/drift-happens-env.sh")
  fi
  if [[ -n "${HOME:-}" ]]; then
    env_candidates+=("$HOME/.config/drift-happens-env.sh")
  fi

  for candidate in "${env_candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      DRIFT_ENV_FILE="$candidate"
      break
    fi
  done
fi

if [[ -z "${DRIFT_ENV_FILE:-}" || ! -f "$DRIFT_ENV_FILE" ]]; then
  echo "Could not find drift-happens env file." >&2
  echo "DRIFT_ENV_FILE=${DRIFT_ENV_FILE:-<unset>}" >&2
  echo "SCRATCH=${SCRATCH:-<unset>}" >&2
  echo "PROJECT_SCRATCH=${PROJECT_SCRATCH:-<unset>}" >&2
  echo "HOME=${HOME:-<unset>}" >&2
  if ((${#env_candidates[@]} > 0)); then
    echo "Searched paths:" >&2
    for candidate in "${env_candidates[@]}"; do
      echo "  $candidate" >&2
    done
  elif [[ -n "${DRIFT_ENV_FILE:-}" ]]; then
    echo "No fallback paths were searched because DRIFT_ENV_FILE was set." >&2
  else
    echo "No fallback paths were available." >&2
  fi
  echo "Set DRIFT_ENV_FILE explicitly or create one at a searched path." >&2
  echo "Example: sbatch --export=ALL,DRIFT_ENV_FILE=/path/to/drift-happens-env.sh ..." >&2
  exit 1
fi
source "$DRIFT_ENV_FILE"

: "${DRIFT_REPO_DIR:=${SCRATCH:?set SCRATCH or DRIFT_REPO_DIR}/git/drift-happens}"
cd "$DRIFT_REPO_DIR"

: "${DRIFT_ARTIFACTS_DIR:?set DRIFT_ARTIFACTS_DIR in DRIFT_ENV_FILE}"

GPU_PLAN_DIR="${DRIFT_GPU_PLAN_DIR:-${DRIFT_ARTIFACTS_DIR}/experiment_plans}"
JOBS_PER_DEVICE="${DRIFT_JOBS_PER_DEVICE:-1}"
PIXI_ENV="${DRIFT_PIXI_ENV:-gpu}"

if [[ -n "${DRIFT_GPU_INDICES:-}" ]]; then
  GPU_INDICES="$DRIFT_GPU_INDICES"
else
  gpu_count="${SLURM_GPUS_ON_NODE:-1}"
  if [[ ! "$gpu_count" =~ ^[0-9]+$ || "$gpu_count" -lt 1 ]]; then
    echo "Warning: SLURM_GPUS_ON_NODE='${SLURM_GPUS_ON_NODE:-}' is invalid or unset; assuming 1 GPU." >&2
    gpu_count=1
  fi
  GPU_INDICES="0"
  for ((gpu_index = 1; gpu_index < gpu_count; gpu_index++)); do
    GPU_INDICES+=",${gpu_index}"
  done
fi

if [[ -n "${DRIFT_SWEEP_CONCURRENCY:-}" ]]; then
  SWEEP_CONCURRENCY="$DRIFT_SWEEP_CONCURRENCY"
else
  IFS="," read -r -a gpu_slots <<< "$GPU_INDICES"
  SWEEP_CONCURRENCY="$((${#gpu_slots[@]} * JOBS_PER_DEVICE))"
fi

if [[ -z "${DRIFT_PLAN_FILE:-}" ]]; then
  : "${DRIFT_PLAN_NAME:?set DRIFT_PLAN_NAME or DRIFT_PLAN_FILE}"
  DRIFT_PLAN_FILE="$GPU_PLAN_DIR/${DRIFT_PLAN_NAME%.yaml}.yaml"
fi

pixi run -e "$PIXI_ENV" materialize
pixi run -e "$PIXI_ENV" drift experiment plans materialize --write \
  --out-dir "$GPU_PLAN_DIR" \
  --device cuda \
  --gpu-indices "$GPU_INDICES" \
  --jobs-per-device "$JOBS_PER_DEVICE" \
  --concurrency "$SWEEP_CONCURRENCY"

pixi run -e "$PIXI_ENV" drift experiment sweep "$DRIFT_PLAN_FILE"
