"""Run the four population-level MACD ablations.

Examples:
    python MACD-ablation.py
    python MACD-ablation.py --ablations MACD-noD MACD-FI --envs Thrower-v0 \
        --num-runs 3 --threads-num 8
"""

import argparse
import json
import os

from macd.run import run
from utils.MyUtils import get_par


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATIONS = {
    "MACD-noM": {"mmse": False, "controller_init": "distill"},
    "MACD-noI": {"mmse": True, "controller_init": "random_init"},
    "MACD-noD": {"mmse": True, "controller_init": "non_distill_copy"},
    "MACD-FI": {"mmse": True, "controller_init": "full_copy"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run MACD population ablations.")
    parser.add_argument("--ablations", nargs="+", choices=tuple(ABLATIONS),
                        default=tuple(ABLATIONS), help="Ablations to run.")
    parser.add_argument("--envs", nargs="+", default=["Carrier-v0", "Thrower-v0"],
                        help="EvoGym environments to evaluate.")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Independent runs per ablation and environment.")
    parser.add_argument("--seed", type=int, default=101,
                        help="Seed for run 0; later runs increment it by one.")
    parser.add_argument("--target-size", type=int, default=5)
    parser.add_argument("--threads-num", type=int, default=20)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--train-iters", type=int, default=64,
                        help="PPO updates per maturity stage (ignored by MACD-noM).")
    parser.add_argument("--promotion-k", type=int, default=None,
                        help="Maximum promoted agents; default is half the population.")
    parser.add_argument("--lambda-max", type=float, default=0.30)
    parser.add_argument("--lambda-min", type=float, default=0.05)
    parser.add_argument("--lambda-tau", type=float, default=2.0)
    parser.add_argument("--selection-eps", type=float, default=1e-8)
    parser.add_argument("--output-root", default=os.path.join(ROOT_DIR, "result", "MACD-ablation"))
    return parser.parse_args()


def validate_args(args):
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be positive")
    if args.threads_num <= 0:
        raise ValueError("--threads-num must be positive")
    if args.pop_size < 2:
        raise ValueError("--pop-size must be at least 2")
    if args.train_iters <= 0:
        raise ValueError("--train-iters must be positive")
    if args.promotion_k is not None and args.promotion_k <= 0:
        raise ValueError("--promotion-k must be positive")


def configure_run(base_args, ablation, env, run_index):
    """Return an independent namespace configured for one ablation run."""
    max_evaluations, total_step = get_par(env)
    policy = ABLATIONS[ablation]
    args = argparse.Namespace(**vars(base_args))
    args.ablation = ablation
    args.env = env
    args.seed = base_args.seed + run_index
    args.mmse = policy["mmse"]
    args.controller_init = policy["controller_init"]
    # Kept for compatibility with downstream code and saved configurations.
    args.distill = args.controller_init == "distill"
    args.total_step = total_step
    args.max_iters = max_evaluations * total_step
    args.train_iters = total_step if ablation == "MACD-noM" else base_args.train_iters
    args.promotion_k = (
        base_args.promotion_k
        if base_args.promotion_k is not None
        else base_args.pop_size // 2
    )
    args.save_to = os.path.join(args.output_root, ablation, env, str(run_index))
    return args


def main():
    args = parse_args()
    validate_args(args)

    for ablation in args.ablations:
        for env in args.envs:
            for run_index in range(args.num_runs):
                run_args = configure_run(args, ablation, env, run_index)
                if os.path.exists(run_args.save_to):
                    print("Skipping existing run: {}".format(run_args.save_to))
                    continue
                os.makedirs(run_args.save_to)
                with open(os.path.join(run_args.save_to, "config.json"), "w") as f:
                    json.dump(vars(run_args), f, indent=4)
                print("Starting {}".format(run_args.save_to))
                run(run_args)


if __name__ == "__main__":
    main()
