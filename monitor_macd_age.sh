#!/usr/bin/env bash
# Start run_MACD_age1.py once run_MACD_age.py has finished.
# Usage: nohup bash monitor_macd_age.sh > monitor_age.log 2>&1 &

set -u

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
LOCK_DIR="$WORKDIR/.macd_age1_started.lock"
PID_FILE="$WORKDIR/age1.pid"
CONDA_SH="/home/dyc/anaconda3/etc/profile.d/conda.sh"

cd "$WORKDIR"

is_running() {
  pgrep -f "[p]ython(3)? .*run_MACD_age\.py([[:space:]]|$)" >/dev/null
}

age1_running() {
  pgrep -f "[p]ython(3)? .*run_MACD_age1\.py([[:space:]]|$)" >/dev/null
}

while is_running; do
  printf '%s run_MACD_age.py is still running; checking again in %ss.\n' \
    "$(date '+%F %T')" "$INTERVAL_SECONDS"
  sleep "$INTERVAL_SECONDS"
done

if age1_running; then
  printf '%s run_MACD_age1.py is already running; nothing to do.\n' "$(date '+%F %T')"
  exit 0
fi

# mkdir is atomic, so concurrent monitors cannot launch the next job twice.
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf '%s age1 launch was already handled (lock: %s).\n' "$(date '+%F %T')" "$LOCK_DIR"
  exit 0
fi

if [[ ! -f "$CONDA_SH" ]]; then
  printf '%s ERROR: Conda initialization script not found: %s\n' "$(date '+%F %T')" "$CONDA_SH" >&2
  rmdir "$LOCK_DIR"
  exit 1
fi

printf '%s run_MACD_age.py has finished; starting run_MACD_age1.py.\n' "$(date '+%F %T')"
source "$CONDA_SH"
conda activate EvoGym
ulimit -n 1048576
nohup python run_MACD_age1.py > age1.log 2>&1 &
echo "$!" > "$PID_FILE"
printf '%s started run_MACD_age1.py (PID %s).\n' "$(date '+%F %T')" "$!"
