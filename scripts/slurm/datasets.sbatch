#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=drift-datasets
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

pixi run datasets-setup yearbook full --yes
pixi run datasets-setup arxiv full --yes
pixi run datasets-setup amazon-reviews-23 full --yes
pixi run datasets-setup imdb-faces full --yes
