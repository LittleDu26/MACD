"""Run the Lamarckian Soft Robot baseline with the MACD benchmark settings.

The experiment budgets and PPO/GA settings are intentionally fixed to the
LamarckianSoftRobot configuration on the research server.  By default all
nine benchmark tasks are run sequentially.  Use ``--task`` to launch one
task, or ``--check`` for a fast installation and environment preflight.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import gym
from evogym import sample_robot

from Lamarckian.experiment_config import ENV_LIST, get_par


ROOT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = ROOT_DIR / "result" / "Lamarckian" / "GA"
SEED = 101
POPULATION_SIZE = 20
PARALLEL_ROBOTS = 20
PPO_ENV_PROCESSES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", *ENV_LIST),
        default="all",
        help="benchmark task to run (default: all tasks, sequentially)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate imports and instantiate every benchmark environment",
    )
    return parser.parse_args()


def preflight(tasks: Tuple[str, ...]) -> None:
    """Check the installed EvoGym stack without starting an experiment."""
    import Lamarckian  # noqa: F401

    body, connections = sample_robot((5, 5))
    for env_name in tasks:
        max_evaluations, max_iters = get_par(env_name)
        env = gym.make(env_name, body=body, connections=connections)
        env.reset()
        env.close()
        print(
            f"OK {env_name}: max_evaluations={max_evaluations}, "
            f"max_iters={max_iters}"
        )


def run_task(env_name: str) -> None:
    max_evaluations, max_iters = get_par(env_name)
    exp_dir = RESULT_ROOT / env_name / f"seed_{SEED}"
    if exp_dir.exists():
        print(f"Resuming {env_name} from {exp_dir}")
        from Lamarckian import Population

        Population.load(exp_dir).evolve()
        return

    print(
        f"Launching {env_name}: max_evaluations={max_evaluations}, "
        f"max_iters={max_iters}, population_size={POPULATION_SIZE}, "
        f"parallel_robots={PARALLEL_ROBOTS}, num_processes={PPO_ENV_PROCESSES}, "
        f"seed={SEED}, cpu_only=True"
    )
    # Population.initialize reads its arguments from sys.argv.  Rebuild that
    # command line so the baseline itself remains unchanged.
    sys.argv = [
        sys.argv[0],
        "--exp-dir", str(exp_dir),
        "--env-name", env_name,
        "--max-evaluations", str(max_evaluations),
        "--max-iters", str(max_iters),
        "--population-size", str(POPULATION_SIZE),
        "--parallel-robots", str(PARALLEL_ROBOTS),
        "--num-processes", str(PPO_ENV_PROCESSES),
        "--no-cuda",
        "--eval-interval", str(max_iters),
        "--seed", str(SEED),
    ]
    from Lamarckian import Population

    Population.initialize().evolve()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tasks = ENV_LIST if args.task == "all" else (args.task,)
    if args.check:
        preflight(tasks)
        return
    for env_name in tasks:
        run_task(env_name)


if __name__ == "__main__":
    main()
