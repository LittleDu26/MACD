"""Run the CuCo baseline with the experiment protocol used on the server.

The default invocation is intentionally equivalent to the server-side
``run_cuco_baseline.py`` launcher: two curriculum stages, CPU, seed 101, a
5x5 target body, and the per-environment update budget below.
"""

import argparse
import math


ENV_SETTINGS = {
    "Walker-v0": (100, 500),
    "AreaMaximizer-v0": (100, 600),
    "Carrier-v0": (100, 500),
    "Thrower-v0": (150, 300),
    "UpStepper-v0": (150, 600),
    "ObstacleTraverser-v0": (150, 1000),
    "ObstacleTraverser-v1": (200, 1000),
    "GapJumper-v0": (200, 1000),
    "BeamSlider-v0": (200, 1000),
}

TIMESTEPS = 2048
ROLLOUT_MULTIPLIER = 128
CURRICULUM_STAGES = 2


def baseline_train_iters(env_name, threads_num):
    """Match the server's CuCo update-budget calculation exactly."""
    max_evaluations, episode_length = ENV_SETTINGS[env_name]
    target_steps_per_stage = (
        max_evaluations * episode_length * ROLLOUT_MULTIPLIER
        / CURRICULUM_STAGES
    )
    return int(math.ceil(target_steps_per_stage / (TIMESTEPS * threads_num)))


def parse_args():
    parser = argparse.ArgumentParser(description="CuCo baseline runner")
    parser.add_argument("--env", choices=ENV_SETTINGS, default="Walker-v0")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--ac_type", choices=("transformer", "fc"), default="transformer")
    parser.add_argument("--target_size", type=int, default=5)
    parser.add_argument("--threads_num", type=int, default=1)
    parser.add_argument("--device_num", type=int, default=-1)
    parser.add_argument("--save_root", default="saved_data")
    parser.add_argument(
        "--single_stage", action="store_true",
        help="Run the CuCo-NCU ablation with the same total two-stage budget.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_iters = baseline_train_iters(args.env, args.threads_num)
    if args.single_stage:
        train_iters *= CURRICULUM_STAGES

    max_evaluations, episode_length = ENV_SETTINGS[args.env]
    print(
        f"env={args.env} max_eva={max_evaluations} tc={episode_length} "
        f"target_size={args.target_size} threads_num={args.threads_num} "
        f"TIMESTEPS={TIMESTEPS} stages={1 if args.single_stage else CURRICULUM_STAGES} "
        f"train_iters_per_stage={train_iters} save_root={args.save_root}"
    )
    if args.dry_run:
        return

    from cuco.run import run

    run(
        env_name=args.env,
        target_design_size=args.target_size,
        seed=args.seed,
        pop_size=args.threads_num,
        train_iters=train_iters,
        ac_type=args.ac_type,
        device_num=args.device_num,
        rl_only=args.single_stage,
        save_root=args.save_root,
    )


if __name__ == "__main__":
    main()
