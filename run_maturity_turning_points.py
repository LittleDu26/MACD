"""Generate fixed random morphologies and record their full PPO learning curves.

The same morphology set is reused for every requested task.  Controllers are
always initialized from scratch; controller checkpoints next to other datasets
are deliberately ignored so task comparisons are not affected by inheritance.

This script is intended to produce calibration curves for choosing non-uniform
maturity boundaries.  It records both raw evaluation returns and running-best
returns at 100 evenly spaced checkpoints over each task's normal full budget.
"""

import argparse
import concurrent.futures
import contextlib
import csv
import hashlib
import json
import multiprocessing
import os
import random
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

import evogym.envs  # noqa: F401 - register EvoGym environments with Gym
import gym
from evogym import get_full_connectivity, hashable, sample_robot

from macd.ppo import PPO
from macd.transformer.config import ppoconfig, transformerconfig
from macd.transformer.transformerPPOagent import PPOAgent, TransformerPPOAC
from utils.MyUtils import eval_robot_constraint


ENV_UPDATES = {
    "Walker-v0": 500,
    "Thrower-v0": 300,
    "GapJumper-v0": 1000,
}
DEFAULT_ENVS = tuple(ENV_UPDATES)
ROBOT_SHAPE = (5, 5)
MAX_PARALLEL_LIMIT = 8
CURVE_FIELDS = (
    "env",
    "robot_index",
    "robot_name",
    "seed",
    "update",
    "env_steps",
    "return",
    "best_return",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fully train a fixed set of random morphologies and save raw and "
            "running-best evaluation curves for maturity turning-point analysis."
        )
    )
    parser.add_argument(
        "--robot-dir",
        default=os.path.join(
            PROJECT_ROOT, "result", "maturity_turning_points", "random_robots_20"
        ),
        help="Directory containing the fixed random morphology set.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            PROJECT_ROOT, "result", "maturity_turning_points", "full_training"
        ),
        help="Directory receiving per-task full learning curves.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        choices=DEFAULT_ENVS,
        default=list(DEFAULT_ENVS),
    )
    parser.add_argument("--num-robots", type=int, default=20)
    parser.add_argument(
        "--robot-seed",
        type=int,
        default=20260905,
        help="Fixed seed used only when generating the morphology set.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=101,
        help=(
            "Shared controller/environment seed for every morphology. Using one "
            "shared seed avoids confounding morphology with different seeds."
        ),
    )
    parser.add_argument(
        "--num-evals",
        type=int,
        default=2,
        help="Evaluation episodes per checkpoint (matches the current MACD default).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=8,
        help="Maximum concurrent training processes; hard-capped at 8.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch/OMP CPU threads used by each training process.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate/verify the morphology set and exit without PPO training.",
    )
    parser.add_argument(
        "--regenerate-robots",
        action="store_true",
        help="Explicitly replace the morphology set using --robot-seed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run tasks that already have a complete learning_curve.csv.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_robots <= 0:
        raise ValueError("--num-robots must be positive")
    if args.num_evals <= 0:
        raise ValueError("--num-evals must be positive")
    if not 1 <= args.max_parallel <= MAX_PARALLEL_LIMIT:
        raise ValueError("--max-parallel must be between 1 and 8")
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    if len(set(args.envs)) != len(args.envs):
        raise ValueError("--envs must not contain duplicates")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def robot_digest(body):
    contiguous = np.ascontiguousarray(body)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def write_json(path, value):
    temp_path = path + ".tmp"
    with open(temp_path, "w") as output_file:
        json.dump(value, output_file, indent=2)
    os.replace(temp_path, path)


def generate_robot_set(robot_dir, count, seed, regenerate=False):
    """Generate unique valid robots with EvoGym's native sampler."""
    manifest_path = os.path.join(robot_dir, "manifest.json")
    if os.path.exists(manifest_path) and not regenerate:
        return load_robot_manifest(robot_dir, count)
    if os.path.exists(robot_dir) and os.listdir(robot_dir) and not regenerate:
        raise RuntimeError(
            "Robot directory is non-empty but has no usable manifest: " + robot_dir
        )

    os.makedirs(robot_dir, exist_ok=True)
    set_all_seeds(seed)
    seen = set()
    records = []
    attempts = 0
    max_attempts = max(1000, count * 1000)
    while len(records) < count and attempts < max_attempts:
        attempts += 1
        body, connections = sample_robot(ROBOT_SHAPE)
        body = np.asarray(body).copy()
        body_key = hashable(body)
        if body_key in seen or not eval_robot_constraint(body):
            continue
        index = len(records)
        name = "robot_{:02d}".format(index)
        filename = name + ".npz"
        np.savez(os.path.join(robot_dir, filename), body, connections)
        seen.add(body_key)
        nonempty = int(np.sum(body != 0))
        records.append(
            {
                "robot_index": index,
                "robot_name": name,
                "filename": filename,
                "sha256": robot_digest(body),
                "shape": list(body.shape),
                "nonempty_voxels": nonempty,
                "actuator_voxels": int(np.sum((body == 3) | (body == 4))),
            }
        )
    if len(records) != count:
        raise RuntimeError(
            "Generated only {} valid unique robots after {} attempts".format(
                len(records), attempts
            )
        )
    manifest = {
        "generator": "evogym.sample_robot",
        "robot_seed": seed,
        "robot_shape": list(ROBOT_SHAPE),
        "num_robots": count,
        "generation_attempts": attempts,
        "robots": records,
    }
    write_json(manifest_path, manifest)
    return manifest


def load_robot_manifest(robot_dir, expected_count):
    manifest_path = os.path.join(robot_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError("Missing robot manifest: " + manifest_path)
    with open(manifest_path) as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("num_robots") != expected_count:
        raise ValueError(
            "Manifest contains {} robots, expected {}".format(
                manifest.get("num_robots"), expected_count
            )
        )
    if len(manifest.get("robots", [])) != expected_count:
        raise ValueError("Robot manifest record count is inconsistent")

    seen = set()
    for expected_index, record in enumerate(manifest["robots"]):
        if record.get("robot_index") != expected_index:
            raise ValueError("Robot manifest indexes must be contiguous from zero")
        path = os.path.join(robot_dir, record["filename"])
        with np.load(path) as data:
            if "arr_0" not in data:
                raise ValueError(path + " does not contain arr_0")
            body = np.asarray(data["arr_0"]).copy()
        if tuple(body.shape) != ROBOT_SHAPE or not eval_robot_constraint(body):
            raise ValueError("Invalid morphology in " + path)
        digest = robot_digest(body)
        if digest != record.get("sha256") or digest in seen:
            raise ValueError("Changed or duplicate morphology in " + path)
        seen.add(digest)
    return manifest


def load_robot_body(path):
    with np.load(path) as data:
        if "arr_0" not in data:
            raise ValueError(path + " does not contain arr_0")
        body = np.asarray(data["arr_0"]).copy()
    if tuple(body.shape) != ROBOT_SHAPE or not eval_robot_constraint(body):
        raise ValueError("Invalid morphology in " + path)
    return body


def get_sample_setting(env_name, body):
    connections = get_full_connectivity(body)
    env = gym.make(
        env_name,
        mode="modular",
        body=body,
        connections=connections,
        env_id=env_name,
    )
    setting = [
        env.modular_state_dim,
        env.modular_action_dim,
        env.other_dim,
        env.voxel_num,
    ]
    env.close()
    return setting


def new_random_controller(sample_setting, ppo_args, trans_args, device):
    actor_critic = TransformerPPOAC(
        modular_state_dim=sample_setting[0],
        modular_action_dim=sample_setting[1],
        sequence_size=sample_setting[3],
        other_feature_size=sample_setting[2],
        ppo_args=ppo_args,
        trans_args=trans_args,
        ac_type="transformer",
        controller_type=trans_args.controller_type,
        device=device,
    )
    return PPOAgent(actor_critic=actor_critic).to(device)


def write_curve_csv(path, records):
    temp_path = path + ".tmp"
    with open(temp_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp_path, path)


def read_curve_csv(path):
    records = []
    with open(path, newline="") as input_file:
        for row in csv.DictReader(input_file):
            records.append(
                {
                    "env": row["env"],
                    "robot_index": int(row["robot_index"]),
                    "robot_name": row["robot_name"],
                    "seed": int(row["seed"]),
                    "update": int(row["update"]),
                    "env_steps": int(row["env_steps"]),
                    "return": float(row["return"]),
                    "best_return": float(row["best_return"]),
                }
            )
    return records


def curve_is_complete(path, env_name):
    if not os.path.isfile(path):
        return False
    updates = ENV_UPDATES[env_name]
    eval_interval = updates // 100
    try:
        records = read_curve_csv(path)
    except (KeyError, TypeError, ValueError):
        return False
    expected_updates = list(range(0, updates + 1, eval_interval))
    return [record["update"] for record in records] == expected_updates


def train_one_robot(
    env_name,
    robot_record,
    robot_dir,
    output_dir,
    training_seed,
    num_evals,
    torch_threads,
):
    """Spawn-process worker for one (environment, morphology) full run."""
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)

    robot_index = robot_record["robot_index"]
    robot_name = robot_record["robot_name"]
    task_dir = os.path.join(output_dir, env_name, robot_name)
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(os.path.join(task_dir, "controllers"), exist_ok=True)
    train_log_path = os.path.join(task_dir, "train.log")
    error_path = os.path.join(task_dir, "stderr.log")
    device = torch.device("cpu")

    try:
        with open(train_log_path, "w") as train_log, open(
            error_path, "w"
        ) as error_log, contextlib.redirect_stdout(train_log), contextlib.redirect_stderr(
            error_log
        ):
            set_all_seeds(training_seed)
            body_path = os.path.join(robot_dir, robot_record["filename"])
            body = load_robot_body(body_path)
            sample_setting = get_sample_setting(env_name, body)

            trans_args = transformerconfig()
            if trans_args.controller_type == "daab":
                trans_args.attention_heads = trans_args.daab_attention_heads
                trans_args.condition_decoder = True
            else:
                trans_args.use_separate_pos_embedding = False

            updates = ENV_UPDATES[env_name]
            eval_interval = updates // 100
            ppo_args = ppoconfig()
            ppo_args.env_name = env_name
            ppo_args.seed = training_seed
            ppo_args.eval_interval = eval_interval
            ppo_args.num_evals = num_evals

            set_all_seeds(training_seed)
            controller = new_random_controller(
                sample_setting, ppo_args, trans_args, device
            )
            controller.ac.device = device
            ppo = PPO(
                robot=(body, get_full_connectivity(body)),
                ppo_size=1,
                train_iters=updates,
                agent=controller,
                verbose=True,
                ppo_args=ppo_args,
                total_step=updates,
                save_path=task_dir,
                device=device,
                mmse=False,
            )
            ppo.train("best", 0, float("-inf"))

            records = [
                {
                    "env": env_name,
                    "robot_index": robot_index,
                    "robot_name": robot_name,
                    "seed": training_seed,
                    "update": record["update"],
                    "env_steps": record["env_steps"],
                    "return": record["return"],
                    "best_return": record["best_return"],
                }
                for record in ppo.evaluation_records
            ]
            expected_updates = list(range(0, updates + 1, eval_interval))
            if [record["update"] for record in records] != expected_updates:
                raise AssertionError(
                    "Curve checkpoints differ from {}".format(expected_updates)
                )
            write_curve_csv(os.path.join(task_dir, "learning_curve.csv"), records)
            write_json(
                os.path.join(task_dir, "metadata.json"),
                {
                    "env": env_name,
                    "robot": robot_record,
                    "controller_initialization": "random",
                    "training_seed": training_seed,
                    "updates": updates,
                    "eval_interval": eval_interval,
                    "num_evals": num_evals,
                    "torch_threads": torch_threads,
                },
            )
        if os.path.getsize(error_path) == 0:
            os.remove(error_path)
        return {
            "status": "completed",
            "env": env_name,
            "robot_index": robot_index,
            "robot_name": robot_name,
            "task_dir": task_dir,
        }
    except Exception:
        with open(error_path, "a") as error_file:
            traceback.print_exc(file=error_file)
        return {
            "status": "failed",
            "env": env_name,
            "robot_index": robot_index,
            "robot_name": robot_name,
            "task_dir": task_dir,
            "error": traceback.format_exc(),
        }


def collect_outputs(output_dir, envs, robot_records):
    all_records = []
    summaries = []
    for env_name in envs:
        for robot_record in robot_records:
            curve_path = os.path.join(
                output_dir,
                env_name,
                robot_record["robot_name"],
                "learning_curve.csv",
            )
            if not curve_is_complete(curve_path, env_name):
                continue
            records = read_curve_csv(curve_path)
            all_records.extend(records)
            final = records[-1]
            summaries.append(
                {
                    "env": env_name,
                    "robot_index": robot_record["robot_index"],
                    "robot_name": robot_record["robot_name"],
                    "seed": final["seed"],
                    "updates": final["update"],
                    "final_return": final["return"],
                    "best_return": final["best_return"],
                }
            )

    if all_records:
        write_curve_csv(os.path.join(output_dir, "all_learning_curves.csv"), all_records)
    if summaries:
        summary_path = os.path.join(output_dir, "summary.csv")
        temp_path = summary_path + ".tmp"
        with open(temp_path, "w", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=(
                    "env",
                    "robot_index",
                    "robot_name",
                    "seed",
                    "updates",
                    "final_return",
                    "best_return",
                ),
            )
            writer.writeheader()
            writer.writerows(summaries)
        os.replace(temp_path, summary_path)
    return all_records, summaries


def main():
    args = parse_args()
    validate_args(args)
    manifest = generate_robot_set(
        args.robot_dir,
        args.num_robots,
        args.robot_seed,
        regenerate=args.regenerate_robots,
    )
    print(
        "Verified {} fixed random robots in {}".format(
            len(manifest["robots"]), args.robot_dir
        )
    )
    if args.prepare_only:
        print("Preparation complete; no PPO training was started.")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    config = {
        "robot_dir": os.path.abspath(args.robot_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "envs": args.envs,
        "environment_updates": {
            env_name: ENV_UPDATES[env_name] for env_name in args.envs
        },
        "eval_intervals": {
            env_name: ENV_UPDATES[env_name] // 100 for env_name in args.envs
        },
        "num_robots": args.num_robots,
        "robot_seed": args.robot_seed,
        "training_seed": args.training_seed,
        "num_evals": args.num_evals,
        "max_parallel": args.max_parallel,
        "torch_threads": args.torch_threads,
        "controller_initialization": "random",
    }
    write_json(os.path.join(args.output_dir, "config.json"), config)

    tasks = []
    statuses = []
    for env_name in args.envs:
        for robot_record in manifest["robots"]:
            curve_path = os.path.join(
                args.output_dir,
                env_name,
                robot_record["robot_name"],
                "learning_curve.csv",
            )
            if not args.force and curve_is_complete(curve_path, env_name):
                statuses.append(
                    {
                        "status": "skipped_complete",
                        "env": env_name,
                        "robot_index": robot_record["robot_index"],
                        "robot_name": robot_record["robot_name"],
                    }
                )
            else:
                tasks.append((env_name, robot_record))

    print(
        "Scheduling {} tasks with at most {} parallel workers.".format(
            len(tasks), args.max_parallel
        )
    )
    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_parallel, mp_context=mp_context
    ) as executor:
        futures = [
            executor.submit(
                train_one_robot,
                env_name,
                robot_record,
                args.robot_dir,
                args.output_dir,
                args.training_seed,
                args.num_evals,
                args.torch_threads,
            )
            for env_name, robot_record in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            statuses.append(result)
            print(
                "[{status}] {env} {robot_name}".format(**result),
                flush=True,
            )
            write_json(
                os.path.join(args.output_dir, "tasks_status.json"), statuses
            )

    all_records, summaries = collect_outputs(
        args.output_dir, args.envs, manifest["robots"]
    )
    failures = [status for status in statuses if status["status"] == "failed"]
    print(
        "Collected {} curve rows and {} completed task summaries.".format(
            len(all_records), len(summaries)
        )
    )
    if failures:
        print("{} tasks failed; inspect their stderr.log files.".format(len(failures)))
        return 1
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
