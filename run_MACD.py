"""Launch MACD experiments with the paper's default benchmark settings."""

import argparse
import copy
import json
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_ENVS = (
    "Walker-v0",
    "AreaMaximizer-v0",
    "Carrier-v0",
    "Thrower-v0",
    "UpStepper-v0",
    "ObstacleTraverser-v0",
    "ObstacleTraverser-v1",
    "GapJumper-v0",
    "BeamSlider-v0",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MACD on one paper task or the complete nine-task benchmark."
    )
    parser.add_argument(
        "--env",
        choices=(*PAPER_ENVS, "all"),
        default="Walker-v0",
        help="task to run, or 'all' for the nine paper tasks (default: Walker-v0)",
    )
    parser.add_argument("--seed", type=int, default=101, help="random seed")
    parser.add_argument(
        "--target_size", type=int, default=5, help="square design-space size"
    )
    parser.add_argument(
        "--threads_num", type=int, default=20, help="number of CPU worker processes"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=None,
        help="global PPO-update budget; defaults to task evaluations x total_step",
    )
    parser.add_argument("--pop_size", type=int, default=20, help="population size")
    parser.add_argument(
        "--train_iters",
        type=int,
        default=64,
        help="PPO updates per maturity stage",
    )
    parser.add_argument(
        "--total_step",
        type=int,
        default=None,
        help="maximum PPO updates per robot; defaults to the paper task setting",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default="",
        help="optional result root; task and run directories are appended",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="experiment name suffix, e.g. XX -> MACD(XX)",
    )
    parser.add_argument("--distill", action="store_true", default=True)
    parser.add_argument("--mmse", action="store_true", default=True)
    parser.add_argument(
        "--promotion_k",
        type=int,
        default=10,
        help="maximum active morphologies promoted each generation",
    )
    parser.add_argument(
        "--lambda_max",
        type=float,
        default=0.30,
        help="initial weight for signed fitness improvement in survivor selection",
    )
    parser.add_argument(
        "--lambda_min",
        type=float,
        default=0.05,
        help="late-stage weight for signed fitness improvement in survivor selection",
    )
    parser.add_argument(
        "--lambda_tau",
        type=float,
        default=2.0,
        help="decay constant for signed improvement weight",
    )
    parser.add_argument(
        "--selection_eps",
        type=float,
        default=1e-8,
        help="numerical epsilon for survivor-selection normalization",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate imports and instantiate the selected environments without training",
    )
    return parser.parse_args()


def selected_environments(env_name):
    return PAPER_ENVS if env_name == "all" else (env_name,)


def paper_task_setting(env_name):
    from utils.MyUtils import get_par

    max_evaluations, max_updates = get_par(env_name)
    if max_evaluations <= 0 or max_updates <= 0:
        raise ValueError("No benchmark setting is defined for {}".format(env_name))
    return max_evaluations, max_updates


def preflight(env_names, target_size):
    import gym
    import torch
    import evogym.envs  # noqa: F401 - registers the environments with Gym
    from evogym import get_full_connectivity, sample_robot

    body, _ = sample_robot((target_size, target_size))
    connections = get_full_connectivity(body)
    for env_name in env_names:
        max_evaluations, max_updates = paper_task_setting(env_name)
        env = gym.make(
            env_name,
            mode="modular",
            body=body,
            connections=connections,
            env_id=env_name,
        )
        observation = env.reset()
        env.close()
        if not isinstance(observation, dict):
            raise TypeError(
                "Expected a modular observation dictionary for {}, got {}".format(
                    env_name, type(observation).__name__
                )
            )
        print(
            "OK {}: evaluations={}, max_updates_per_robot={}".format(
                env_name, max_evaluations, max_updates
            )
        )
    print("Environment check passed (PyTorch {}, CPU mode).".format(torch.__version__))


def default_result_root(args):
    method_name = "MACD"
    if not args.mmse:
        method_name += "_noM"
    if not args.distill:
        method_name += "_noS"
    if args.suffix:
        method_name += "({})".format(args.suffix)
    return os.path.join(ROOT_DIR, "result", method_name)


def main():
    args = parse_args()
    env_names = selected_environments(args.env)

    if args.check:
        preflight(env_names, args.target_size)
        return

    from macd.run import run

    result_root = args.save_to or default_result_root(args)
    for env_name in env_names:
        max_evaluations, paper_max_updates = paper_task_setting(env_name)
        run_args = copy.deepcopy(args)
        run_args.env = env_name
        run_args.total_step = (
            args.total_step if args.total_step is not None else paper_max_updates
        )
        run_args.max_iters = (
            args.max_iters
            if args.max_iters is not None
            else max_evaluations * run_args.total_step
        )
        run_args.save_to = os.path.join(result_root, env_name, "0")

        if os.path.exists(run_args.save_to):
            print("Skipping existing run directory: {}".format(run_args.save_to))
            continue

        os.makedirs(run_args.save_to)
        with open(os.path.join(run_args.save_to, "config.json"), "w") as stream:
            json.dump(vars(run_args), stream, indent=4)
        run(run_args)


if __name__ == "__main__":
    main()
