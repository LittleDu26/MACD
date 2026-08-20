#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_EXPERIMENT_DIR=${SOURCE_EXPERIMENT_DIR:-"${ROOT_DIR}/result/attention_loss_ablation_stratified_20"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"${ROOT_DIR}/result/controller_copy_distill_ablation_stratified_20"}
MAX_PARALLEL=${MAX_PARALLEL:-40}
TORCH_THREADS=${TORCH_THREADS:-1}
MODE=${1:-train}

train_experiment() {
  "${PYTHON_BIN}" "${ROOT_DIR}/run_controller_copy_distill_ablation_tasks.py" \
    --offspring-dir "${SOURCE_EXPERIMENT_DIR}/offspring" \
    --results-dir "${EXPERIMENT_DIR}/results" \
    --updates 200 \
    --eval-interval 5 \
    --num-evals 2 \
    --seed-base 2000 \
    --max-parallel "${MAX_PARALLEL}" \
    --torch-threads "${TORCH_THREADS}"
}

case "${MODE}" in
  train)
    train_experiment
    ;;
  *)
    echo "Usage: $0 [train]" >&2
    exit 2
    ;;
esac
