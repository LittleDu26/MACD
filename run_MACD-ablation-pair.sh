#!/usr/bin/env bash
# Run MACD ablation environment suites in pairs. Each suite keeps its own
# 20-worker training pool and runs MACD-noM, MACD-noI, MACD-noD, MACD-FI
# sequentially. Edit TASKS to choose the environments to run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
THREADS_PER_TASK="${THREADS_PER_TASK:-20}"

# Edit this list as needed. The script launches two environments at a time,
# then waits for both to finish before taking the next pair.
TASKS=(
    "Walker-v0"
    "Thrower-v0"
    "ObstacleTraverser-v1"
    "GapJumper-v0"
)
ACTIVE_PIDS=()

stop_active_tasks() {
    if (( ${#ACTIVE_PIDS[@]} > 0 )); then
        kill -TERM "${ACTIVE_PIDS[@]}" 2>/dev/null || true
    fi
}
trap stop_active_tasks INT TERM

run_task() {
    local env_name="$1"
    local log_file="$SCRIPT_DIR/result/MACD-ablation-${env_name}.log"
    mkdir -p "$(dirname "$log_file")"
    echo "Starting ${env_name} with ${THREADS_PER_TASK} threads; log: ${log_file}"
    "$PYTHON_BIN" "$SCRIPT_DIR/run_MACD-ablation.py" \
        --envs "$env_name" \
        --threads-num "$THREADS_PER_TASK" \
        >"$log_file" 2>&1
}

if (( ${#TASKS[@]} == 0 )); then
    echo "TASKS is empty; add at least one environment." >&2
    exit 2
fi

run_pair() {
    local first_env="$1"
    local second_env="${2:-}"
    local first_pid second_pid first_status=0 second_status=0

    run_task "$first_env" &
    first_pid=$!
    ACTIVE_PIDS=("$first_pid")
    if [[ -n "$second_env" ]]; then
        run_task "$second_env" &
        second_pid=$!
        ACTIVE_PIDS+=("$second_pid")
    fi

    if wait "$first_pid"; then :; else first_status=$?; fi
    if [[ -n "$second_env" ]]; then
        if wait "$second_pid"; then :; else second_status=$?; fi
    fi
    if (( first_status != 0 || second_status != 0 )); then
        echo "Task pair failed: ${first_env}=${first_status}, ${second_env:-none}=${second_status}" >&2
        exit 1
    fi
    ACTIVE_PIDS=()
}

for ((index = 0; index < ${#TASKS[@]}; index += 2)); do
    first_env="${TASKS[index]}"
    second_env="${TASKS[index + 1]:-}"
    echo "Running pair: ${first_env}${second_env:+ and ${second_env}}"
    run_pair "$first_env" "$second_env"
done
