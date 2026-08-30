#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_RESULT_DIR=${SOURCE_RESULT_DIR:-"${ROOT_DIR}/result/MACD(2loss)/Walker-v0/0"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"${ROOT_DIR}/result/attention_loss_ablation_stratified_20"}
MAX_PARALLEL=${MAX_PARALLEL:-0}
TORCH_THREADS=${TORCH_THREADS:-0}
MODE=${1:-train}

prepare_experiment() {
  "${PYTHON_BIN}" "${ROOT_DIR}/tests/prepare_loss_ablation.py" \
    --result-dir "${SOURCE_RESULT_DIR}" \
    --output-dir "${EXPERIMENT_DIR}" \
    --min-parent-maturity 13 \
    --parent-pool-size 30 \
    --distance-bins 1-5 6-10 11-15 16-20 \
    --offspring-per-bin 5 \
    --sampling-seed 20260818 \
    --mutation-seed-base 1000 \
    --max-chain-steps 30 \
    --max-mutation-retries 100
}

train_experiment() {
  "${PYTHON_BIN}" "${ROOT_DIR}/tests/run_loss_ablation_tasks.py" \
    --offspring-dir "${EXPERIMENT_DIR}/offspring" \
    --results-dir "${EXPERIMENT_DIR}/results" \
    --updates 200 \
    --eval-interval 5 \
    --num-evals 2 \
    --seed-base 2000 \
    --max-parallel "${MAX_PARALLEL}" \
    --torch-threads "${TORCH_THREADS}"
}

case "${MODE}" in
  prepare)
    prepare_experiment
    ;;
  train)
    train_experiment
    ;;
  all)
    prepare_experiment
    train_experiment
    ;;
  *)
    echo "Usage: $0 [prepare|train|all]" >&2
    exit 2
    ;;
esac
