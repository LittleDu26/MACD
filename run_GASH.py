"""Launch the remote EvoGym-GASH SuHaGA baseline from this project root.

The defaults below are the exact arguments from the remote project's
``run_ga_tasks.sh`` for its default BridgeWalker-v0 task.  Supply command-line
arguments to override them, using the same option names as the remote script.
Results are kept under ``result/GASH`` unless ``EVOGYM_GASH_SAVED_DATA`` is set.
"""

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
GASH_ROOT = PROJECT_ROOT / "GASH"

REMOTE_DEFAULT_ARGS = [
    "--algo", "ppo",
    "--use-gae",
    "--lr", "2.5e-4",
    "--clip-param", "0.1",
    "--value-loss-coef", "0.5",
    "--num-processes", "1",
    "--num-steps", "128",
    "--num-mini-batch", "4",
    "--log-interval", "512",
    "--use-linear-lr-decay",
    "--entropy-coef", "0.01",
    "--eval-interval", "64",
    "--randseed", "101",
    "--is-pruning", "1",
    "--env-name-for-ist", "BridgeWalker-v0",
    "--ga-num-cores", "20",
    "--budget-termination", "1",
]


def main() -> None:
    # The remote utility resolves this relative to GASH_ROOT, hence "../result".
    os.environ.setdefault("EVOGYM_GASH_SAVED_DATA", "../result/GASH")
    sys.path.insert(0, str(GASH_ROOT))
    if len(sys.argv) == 1:
        sys.argv.extend(REMOTE_DEFAULT_ARGS)
    runpy.run_module("entrypoint", run_name="__main__")


if __name__ == "__main__":
    main()
