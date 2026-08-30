"""Early controller-learning ablation for one mutated Walker-v0 offspring.

This experiment deliberately bypasses MACD population selection.  It creates
one offspring with MACD's mutation operator and trains four controllers on
that same morphology: random initialization, attention distillation without
inheritance, parameter inheritance without attention warm-up, and inheritance
with shared-voxel attention distillation.
"""

import argparse
import copy
import csv
import concurrent.futures
import importlib
import json
import multiprocessing
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import evogym.envs  # noqa: F401 - registers EvoGym environments with Gym
import gym
from evogym import get_full_connectivity, hashable, has_actuator, is_connected

from macd.controller_distillation import (
    attention_distill_warmup,
    build_child_controller,
    collect_parent_rollout_obs,
    same_voxel_mask,
    should_skip_warmup,
)
from macd.ppo import PPO
from macd.transformer.config import ppoconfig, transformerconfig
from macd.transformer.transformerPPOagent import PPOAgent, TransformerPPOAC
from utils.algo_utils import mutate


ROOT_DIR = PROJECT_ROOT
ARM_ORDER = (
    "random_init",
    "distill_only",
    "inherit_only",
    "inherit_distill",
)
ARM_LABELS = {
    "random_init": "Random initialization",
    "distill_only": "Shared-voxel distillation only",
    "inherit_only": "Inheritance only",
    "inherit_distill": "Inheritance + shared-voxel distillation",
}
ARM_DIRECTORIES = {
    "random_init": "MACD_noS",
    "inherit_only": "MACD_noD",
    "inherit_distill": "MACD",
    "distill_only": "MACD_noC",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train early learning curves for a fixed mutated offspring."
    )
    parser.add_argument(
        "--parent-structure",
        default=os.path.join(ROOT_DIR, "result", "422.npz"),
        help="Parent morphology .npz file.",
    )
    parser.add_argument(
        "--parent-controller",
        default=os.path.join(ROOT_DIR, "result", "422.pt"),
        help="Parent controller checkpoint.",
    )
    parser.add_argument("--env", default="Walker-v0")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run one seed (defaults to 101 when neither seed option is given).",
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run multiple independent seeds, for example: --seeds 102 103",
    )
    parser.add_argument(
        "--offspring-seed",
        type=int,
        default=101,
        help="Seed used once to create the shared offspring (default: 101).",
    )
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--num-evals", type=int, default=2)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARM_ORDER,
        default=None,
        help=(
            "Controller arms to train. Defaults to all four; use "
            "--arms distill_only to supplement existing result directories."
        ),
    )
    parser.add_argument("--max-mutation-retries", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--max-parallel-tasks",
        type=int,
        default=4,
        help="Maximum concurrent (seed, arm) training processes; capped at 4.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Single-seed output directory, or the parent directory for "
            "seed_<seed> folders in multi-seed mode."
        ),
    )
    parser.add_argument(
        "--shared-offspring-dir",
        default=None,
        help=(
            "Directory containing the one shared offspring. Defaults to "
            "<experiment-root>/shared_offspring."
        ),
    )
    parser.add_argument(
        "--regenerate-offspring",
        action="store_true",
        help="Explicitly replace an existing shared offspring.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.updates <= 0:
        raise ValueError("--updates must be positive")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive")
    if args.updates % args.eval_interval != 0:
        raise ValueError("--updates must be divisible by --eval-interval")
    if args.num_evals <= 0:
        raise ValueError("--num-evals must be positive")
    if args.max_mutation_retries <= 0:
        raise ValueError("--max-mutation-retries must be positive")
    if args.arms is not None and len(set(args.arms)) != len(args.arms):
        raise ValueError("Arm names must be unique")
    if not 1 <= args.max_parallel_tasks <= 4:
        raise ValueError("--max-parallel-tasks must be between 1 and 4")
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def register_legacy_checkpoint_aliases():
    """Map the checkpoint's historical ``agcd`` modules to current ``macd``."""
    aliases = {
        "agcd": "macd",
        "agcd.transformer": "macd.transformer",
        "agcd.transformer.transformerPPOagent":
            "macd.transformer.transformerPPOagent",
        "agcd.transformer.transformermodel":
            "macd.transformer.transformermodel",
        "agcd.transformer.transformer": "macd.transformer.transformer",
        "agcd.transformer.config": "macd.transformer.config",
        "agcd.transformer.distributions": "macd.transformer.distributions",
        "agcd.envs": "macd.envs",
    }
    for old_name, current_name in aliases.items():
        sys.modules[old_name] = importlib.import_module(current_name)


def load_parent_structure(path):
    with np.load(path) as data:
        if "arr_0" not in data:
            raise ValueError("Parent structure must contain arr_0 (the voxel body)")
        body = np.asarray(data["arr_0"]).copy()
    if body.ndim != 2:
        raise ValueError("Parent voxel body must be a 2-D array")
    return body


def load_parent_controller(path, device):
    register_legacy_checkpoint_aliases()
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, (list, tuple)) or len(checkpoint) != 2:
        raise ValueError("Expected controller checkpoint [controller, obs_rms]")
    controller, obs_rms = checkpoint
    controller.to(device)
    controller.ac.device = device
    # prepare_distilled_controller historically looks for this attached field.
    controller.obs_rms = copy.deepcopy(obs_rms)
    return controller, obs_rms


def generate_offspring(parent_body, seed, max_retries):
    set_all_seeds(seed)
    record = {hashable(parent_body): []}
    for attempt in range(1, max_retries + 1):
        child = mutate(parent_body, record)
        if child is None:
            continue
        if np.array_equal(child, parent_body):
            continue
        if not is_connected(child) or not has_actuator(child):
            continue
        if should_skip_warmup(parent_body, child):
            # A distillation arm that silently skips warm-up is not informative.
            continue
        return np.asarray(child).copy(), attempt
    raise RuntimeError(
        "Could not generate a valid offspring with at least four shared voxels "
        "after {} mutation calls".format(max_retries)
    )


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
        device=device,
    )
    return PPOAgent(actor_critic=actor_critic).to(device)


def state_dicts_equal(left, right):
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def changed_parameter_names(before_state, controller):
    after_state = controller.state_dict()
    return [
        name
        for name, before_value in before_state.items()
        if not torch.equal(before_value, after_state[name])
    ]


def qk_slices_match(random_controller, inherited_controller):
    random_state = random_controller.state_dict()
    inherited_state = inherited_controller.state_dict()
    for name, random_value in random_state.items():
        if name.endswith("self_attn.in_proj_weight") or name.endswith(
            "self_attn.in_proj_bias"
        ):
            embed_dim = random_value.shape[0] // 3
            if not torch.equal(
                random_value[: 2 * embed_dim],
                inherited_state[name][: 2 * embed_dim],
            ):
                return False
    return True


def prepare_controller_arms(
    parent_controller,
    parent_obs_rms,
    parent_body,
    child_body,
    sample_setting,
    ppo_args,
    trans_args,
    env_name,
    seed,
    device,
):
    # Resetting the seed before both constructors aligns the random baseline's
    # Q/K values with the Q/K values restored during parameter inheritance.
    set_all_seeds(seed)
    random_controller = new_random_controller(
        sample_setting, ppo_args, trans_args, device
    )
    set_all_seeds(seed)
    inherit_controller = build_child_controller(
        parent_controller, sample_setting, ppo_args, trans_args, device
    )
    if not qk_slices_match(random_controller, inherit_controller):
        raise AssertionError("Random and inherited Q/K initializations do not match")

    distill_only_controller = copy.deepcopy(random_controller)
    inherited_distill_controller = copy.deepcopy(inherit_controller)
    if not state_dicts_equal(random_controller, distill_only_controller):
        raise AssertionError(
            "Random and distillation-only arms differ before attention warm-up"
        )
    if not state_dicts_equal(inherit_controller, inherited_distill_controller):
        raise AssertionError(
            "Inheritance arms differ before attention warm-up"
        )

    before_distill_only = {
        name: value.detach().clone()
        for name, value in distill_only_controller.state_dict().items()
    }
    before_inherited_distill = {
        name: value.detach().clone()
        for name, value in inherited_distill_controller.state_dict().items()
    }
    set_all_seeds(seed)
    observations = collect_parent_rollout_obs(
        parent_controller,
        parent_body,
        env_name,
        seed,
        device,
        ob_rms=parent_obs_rms,
    )
    set_all_seeds(seed)
    attention_distill_warmup(
        parent_controller,
        distill_only_controller,
        observations,
        parent_body,
        child_body,
        device,
        loss_type=trans_args.attention_distill_loss,
        lambda_attention=trans_args.attention_distill_lambda_a,
        lambda_feature=trans_args.attention_distill_lambda_h,
    )
    set_all_seeds(seed)
    attention_distill_warmup(
        parent_controller,
        inherited_distill_controller,
        observations,
        parent_body,
        child_body,
        device,
        loss_type=trans_args.attention_distill_loss,
        lambda_attention=trans_args.attention_distill_lambda_a,
        lambda_feature=trans_args.attention_distill_lambda_h,
    )
    warmup_changes = {
        "distill_only": changed_parameter_names(
            before_distill_only, distill_only_controller
        ),
        "inherit_distill": changed_parameter_names(
            before_inherited_distill, inherited_distill_controller
        ),
    }
    for arm, changed_names in warmup_changes.items():
        if not changed_names:
            raise AssertionError(
                "Attention warm-up did not change any parameters for {}".format(arm)
            )

    return {
        "random_init": random_controller,
        "distill_only": distill_only_controller,
        "inherit_only": inherit_controller,
        "inherit_distill": inherited_distill_controller,
    }, len(observations), warmup_changes


def make_arm_directories(output_dir):
    for arm in ARM_ORDER:
        arm_dir = os.path.join(output_dir, ARM_DIRECTORIES[arm])
        controllers_dir = os.path.join(arm_dir, "controllers")
        os.makedirs(controllers_dir, exist_ok=True)


def train_arm(
    arm,
    controller,
    child_body,
    args,
    ppo_args,
    output_dir,
    device,
):
    set_all_seeds(args.seed)
    controller.to(device)
    controller.ac.device = device
    arm_dir = os.path.join(output_dir, ARM_DIRECTORIES[arm])
    ppo = PPO(
        robot=(child_body, get_full_connectivity(child_body)),
        ppo_size=1,
        train_iters=args.updates,
        agent=controller,
        verbose=True,
        ppo_args=ppo_args,
        total_step=args.updates,
        save_path=arm_dir,
        device=device,
        mmse=False,
    )
    ppo.train("best", 0, float("-inf"))

    records = []
    for record in ppo.evaluation_records:
        row = {
            "seed": args.seed,
            "env": args.env,
            "arm": arm,
            "update": record["update"],
            "env_steps": record["env_steps"],
            "return": record["return"],
            "best_return": record["best_return"],
        }
        records.append(row)
    expected_count = 1 + args.updates // args.eval_interval
    if len(records) != expected_count or records[-1]["update"] != args.updates:
        raise AssertionError(
            "{} produced {} curve points; expected {} ending at update {}".format(
                arm, len(records), expected_count, args.updates
            )
        )
    return records


def write_curve_csv(path, records):
    fieldnames = (
        "seed",
        "env",
        "arm",
        "update",
        "env_steps",
        "return",
        "best_return",
    )
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def read_curve_csv(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            records.append(
                {
                    "seed": int(row["seed"]),
                    "env": row["env"],
                    "arm": row["arm"],
                    "update": int(row["update"]),
                    "env_steps": int(row["env_steps"]),
                    "return": float(row["return"]),
                    "best_return": float(row["best_return"]),
                }
            )
    return records


def plot_curves(output_dir, records, metric, stem, ylabel, env_name):
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for arm in ARM_ORDER:
        arm_records = [record for record in records if record["arm"] == arm]
        axis.plot(
            [record["update"] for record in arm_records],
            [record[metric] for record in arm_records],
            linewidth=1.8,
            label=ARM_LABELS[arm],
        )
    axis.set_xlabel("PPO updates")
    axis.set_ylabel(ylabel)
    axis.set_title("{} offspring early controller learning".format(env_name))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            os.path.join(output_dir, "{}.{}".format(stem, extension)),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def morphology_metadata(parent_body, child_body, mutation_attempt):
    changed_positions = np.argwhere(parent_body != child_body)
    changed_voxels = []
    for row, column in changed_positions:
        changed_voxels.append(
            {
                "row": int(row),
                "column": int(column),
                "parent_type": int(parent_body[row, column]),
                "child_type": int(child_body[row, column]),
            }
        )
    shared_positions = np.argwhere(same_voxel_mask(parent_body, child_body).reshape(parent_body.shape))
    return {
        "mutation_call": mutation_attempt,
        "changed_voxel_count": len(changed_voxels),
        "changed_voxels": changed_voxels,
        "shared_nonempty_voxel_count": int(len(shared_positions)),
        "shared_nonempty_positions": [
            [int(row), int(column)] for row, column in shared_positions
        ],
        "parent_body": parent_body.astype(int).tolist(),
        "child_body": child_body.astype(int).tolist(),
    }


def validate_shared_offspring(parent_body, child_body):
    if child_body.shape != parent_body.shape:
        raise ValueError("Shared offspring shape does not match the parent")
    if np.array_equal(child_body, parent_body):
        raise ValueError("Shared offspring is identical to the parent")
    if not is_connected(child_body) or not has_actuator(child_body):
        raise ValueError("Shared offspring is not a valid connected actuator robot")
    if should_skip_warmup(parent_body, child_body):
        raise ValueError("Shared offspring has fewer than four shared nonempty voxels")


def load_or_create_shared_offspring(args, parent_body, shared_dir):
    os.makedirs(shared_dir, exist_ok=True)
    parent_path = os.path.join(shared_dir, "parent.npz")
    offspring_path = os.path.join(shared_dir, "offspring.npz")
    metadata_path = os.path.join(shared_dir, "morphology.json")

    if os.path.exists(offspring_path) and not args.regenerate_offspring:
        if not os.path.exists(parent_path):
            raise ValueError(
                "Shared offspring exists without parent.npz: {}".format(shared_dir)
            )
        saved_parent = load_parent_structure(parent_path)
        if not np.array_equal(saved_parent, parent_body):
            raise ValueError(
                "Shared offspring was generated from a different parent; use "
                "--regenerate-offspring to replace it explicitly"
            )
        child_body = load_parent_structure(offspring_path)
        validate_shared_offspring(parent_body, child_body)
        if os.path.exists(metadata_path):
            with open(metadata_path) as metadata_file:
                metadata = json.load(metadata_file)
        else:
            metadata = morphology_metadata(parent_body, child_body, None)
        print("Reusing shared offspring: {}".format(offspring_path))
        return child_body, metadata, offspring_path

    child_body, mutation_attempt = generate_offspring(
        parent_body, args.offspring_seed, args.max_mutation_retries
    )
    validate_shared_offspring(parent_body, child_body)
    np.savez(parent_path, parent_body, get_full_connectivity(parent_body))
    np.savez(offspring_path, child_body, get_full_connectivity(child_body))
    metadata = morphology_metadata(parent_body, child_body, mutation_attempt)
    metadata["offspring_seed"] = args.offspring_seed
    metadata["parent_structure_path"] = os.path.abspath(args.parent_structure)
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print("Created shared offspring: {}".format(offspring_path))
    return child_body, metadata, offspring_path


def prepare_seed_tasks(args, device, parent_body, child_body,
                       shared_offspring_path):
    validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    parent_controller, parent_obs_rms = load_parent_controller(
        args.parent_controller, device
    )
    sample_setting = get_sample_setting(args.env, parent_body)

    ppo_args = ppoconfig()
    ppo_args.env_name = args.env
    ppo_args.seed = args.seed
    ppo_args.eval_interval = args.eval_interval
    ppo_args.num_evals = args.num_evals
    trans_args = transformerconfig()

    controllers, rollout_observations, warmup_changes = prepare_controller_arms(
        parent_controller,
        parent_obs_rms,
        parent_body,
        child_body,
        sample_setting,
        ppo_args,
        trans_args,
        args.env,
        args.seed,
        device,
    )

    make_arm_directories(args.output_dir)

    metadata = {
        "shared_offspring_path": os.path.abspath(shared_offspring_path),
        "shared_offspring_hash": hashable(child_body),
        "training_seed": args.seed,
    }
    metadata["parent_controller_path"] = os.path.abspath(args.parent_controller)
    metadata["parent_rollout_observations"] = rollout_observations
    metadata["warmup_changed_parameters"] = warmup_changes
    with open(
        os.path.join(args.output_dir, "morphology_and_distillation.json"),
        "w",
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "ppo_timesteps_per_update": int(ppo_args.TIMESTEPS),
            "arms": list(ARM_ORDER),
            "arm_directories": dict(ARM_DIRECTORIES),
            "requested_arms": (
                list(args.arms) if args.arms is not None else list(ARM_ORDER)
            ),
            "shared_offspring_path": os.path.abspath(shared_offspring_path),
            "shared_offspring_hash": hashable(child_body),
            "evolutionary_selection": False,
        }
    )
    with open(os.path.join(args.output_dir, "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=2)

    selected_arms = list(args.arms) if args.arms is not None else list(ARM_ORDER)
    combined_csv_path = os.path.join(args.output_dir, "learning_curves.csv")
    existing_records = read_curve_csv(combined_csv_path)
    retained_records = [
        record for record in existing_records if record["arm"] not in selected_arms
    ]
    tasks = []
    for arm in selected_arms:
        tasks.append(
            (arm, controllers[arm], child_body, args, ppo_args, args.output_dir)
        )

    return {
        "args": args,
        "combined_csv_path": combined_csv_path,
        "retained_records": retained_records,
        "selected_arms": selected_arms,
        "tasks": tasks,
        "completed_records": {},
    }


def train_arm_worker(arm, controller, child_body, args, ppo_args, output_dir):
    """Process-pool entry point for one independent (seed, arm) task."""
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cpu")
    records = train_arm(
        arm,
        controller,
        child_body,
        args,
        ppo_args,
        output_dir,
        device,
    )
    return arm, records


def finalize_seed_tasks(context):
    args = context["args"]
    all_records = list(context["retained_records"])
    for arm in context["selected_arms"]:
        arm_records = context["completed_records"][arm]
        write_curve_csv(
            os.path.join(
                args.output_dir,
                ARM_DIRECTORIES[arm],
                "learning_curve.csv",
            ),
            arm_records,
        )
        all_records.extend(arm_records)

    arm_rank = {arm: index for index, arm in enumerate(ARM_ORDER)}
    all_records.sort(
        key=lambda record: (arm_rank.get(record["arm"], len(ARM_ORDER)),
                            record["update"])
    )
    write_curve_csv(context["combined_csv_path"], all_records)
    plot_curves(
        args.output_dir,
        all_records,
        metric="return",
        stem="learning_curves_raw",
        ylabel="Evaluation return",
        env_name=args.env,
    )
    plot_curves(
        args.output_dir,
        all_records,
        metric="best_return",
        stem="learning_curves_best_so_far",
        ylabel="Best-so-far evaluation return",
        env_name=args.env,
    )
    print("Experiment outputs written to {}".format(args.output_dir))


def main():
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cpu")

    if args.seeds is not None:
        seeds = list(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [101]

    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed values must be unique")

    requested_output_dir = args.output_dir
    default_experiment_root = os.path.join(
        ROOT_DIR,
        "result",
        "offspring_attention_ablation",
        args.env,
    )
    if requested_output_dir is None:
        experiment_root = default_experiment_root
    else:
        experiment_root = os.path.abspath(requested_output_dir)

    if args.shared_offspring_dir is None:
        shared_offspring_dir = os.path.join(
            experiment_root, "shared_offspring"
        )
    else:
        shared_offspring_dir = os.path.abspath(args.shared_offspring_dir)

    parent_body = load_parent_structure(args.parent_structure)
    child_body, _, shared_offspring_path = load_or_create_shared_offspring(
        args, parent_body, shared_offspring_dir
    )

    seed_contexts = []
    for seed in seeds:
        seed_args = copy.deepcopy(args)
        seed_args.seed = seed
        seed_args.seeds = list(seeds)
        seed_args.shared_offspring_dir = shared_offspring_dir
        if requested_output_dir is None:
            seed_args.output_dir = os.path.join(
                experiment_root,
                "seed_{}".format(seed),
            )
        elif len(seeds) == 1:
            seed_args.output_dir = requested_output_dir
        else:
            seed_args.output_dir = os.path.join(
                requested_output_dir, "seed_{}".format(seed)
            )
        seed_args.output_dir = os.path.abspath(seed_args.output_dir)

        print("\n=== Preparing seed {} ===".format(seed))
        context = prepare_seed_tasks(
            seed_args,
            device,
            parent_body,
            child_body,
            shared_offspring_path,
        )
        seed_contexts.append(context)

    total_tasks = sum(len(context["tasks"]) for context in seed_contexts)
    worker_count = min(args.max_parallel_tasks, total_tasks)
    print(
        "\nRunning {} controller tasks with up to {} concurrent processes...".format(
            total_tasks, worker_count
        )
    )
    spawn_context = multiprocessing.get_context("spawn")
    future_context = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=spawn_context,
    ) as executor:
        for context in seed_contexts:
            for task in context["tasks"]:
                future = executor.submit(train_arm_worker, *task)
                future_context[future] = context

        for future in concurrent.futures.as_completed(future_context):
            context = future_context[future]
            arm, records = future.result()
            context["completed_records"][arm] = records
            print(
                "Completed seed {} / {}".format(context["args"].seed, arm),
                flush=True,
            )

    for context in seed_contexts:
        finalize_seed_tasks(context)


if __name__ == "__main__":
    main()
