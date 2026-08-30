"""Run the stratified controller-copy and distillation ablation.

The experiment reuses prepared offspring and compares four controller starts:
full parent copy, full copy plus shared-attention-feature distillation, copying
only non-distillation parameters, and complete random initialization.
"""

import argparse
import concurrent.futures
import copy
import csv
import json
import multiprocessing
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from evogym import get_full_connectivity

import tests.run_loss_ablation_tasks as loss_runner
import tests.run_offspring_attention_ablation as roaa
from macd.controller_distillation import (
    _is_layernorm_parameter,
    _is_separate_qk_parameter,
    attention_distill_warmup,
    build_copied_child_controller,
    collect_parent_rollout_obs,
)
from macd.ppo import PPO
from macd.transformer.config import ppoconfig, transformerconfig


ROOT_DIR = PROJECT_ROOT
ARM_ORDER = (
    "full_copy_no_distill",
    "full_copy_distill",
    "non_distill_copy_no_distill",
    "random_init",
)
ARM_LABELS = {
    "full_copy_no_distill": "Full copy, no distillation",
    "full_copy_distill": "Full copy + distillation",
    "non_distill_copy_no_distill": "Non-distillation copy",
    "random_init": "Random initialization",
}
ARM_COPY_MODES = {
    "full_copy_no_distill": "full_copy",
    "full_copy_distill": "full_copy",
    "non_distill_copy_no_distill": "non_distill_copy",
    "random_init": "random_init",
}
DISTILL_ARM = "full_copy_distill"
DISTILL_LOSS = "attention_kl_feature_mse"
CURVE_FIELDS = (
    "offspring_index",
    "arm",
    "seed",
    "update",
    "env_steps",
    "return",
    "best_return",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the stratified controller copy/distillation ablation."
    )
    parser.add_argument(
        "--offspring-dir",
        default=os.path.join(
            ROOT_DIR,
            "result",
            "attention_loss_ablation_stratified_20",
            "offspring",
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(
            ROOT_DIR,
            "result",
            "controller_copy_distill_ablation_stratified_20",
            "results",
        ),
    )
    parser.add_argument("--env", default="Walker-v0")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--num-evals", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Concurrent worker processes; 0 auto-detects (maximum 40).",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Torch/OMP/MKL threads per worker; 0 auto-detects.",
    )
    parser.add_argument(
        "--offspring-ids",
        type=int,
        nargs="+",
        default=None,
        help="Restrict execution to selected offspring indexes.",
    )
    parser.add_argument(
        "--only-arm",
        choices=ARM_ORDER,
        default=None,
        help="Restrict execution to one arm for smoke tests or recovery.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run tasks whose learning_curve.csv is already complete.",
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
    if not 0 <= args.max_parallel <= loss_runner.MAX_PARALLEL_LIMIT:
        raise ValueError("--max-parallel must be between 0 and 40")
    if args.torch_threads < 0:
        raise ValueError("--torch-threads must be nonnegative")


def clone_state(controller):
    return {
        name: value.detach().clone()
        for name, value in controller.state_dict().items()
    }


def is_combined_qkv(name):
    return name.endswith("self_attn.in_proj_weight") or name.endswith(
        "self_attn.in_proj_bias"
    )


def is_distillation_state_entry(name):
    return (
        is_combined_qkv(name)
        or _is_separate_qk_parameter(name)
        or _is_layernorm_parameter(name)
    )


def audit_full_copy(parent_state, child_state):
    if parent_state.keys() != child_state.keys():
        raise AssertionError("Full-copy controller state keys differ from parent")
    mismatched = [
        name
        for name in parent_state
        if not torch.equal(parent_state[name], child_state[name])
    ]
    if mismatched:
        raise AssertionError(
            "Full-copy controller differs from parent: {}".format(mismatched)
        )
    return {
        "copied_parameter_entries": list(parent_state.keys()),
        "random_parameter_entries": [],
    }


def audit_non_distill_copy(parent_state, random_state, child_state):
    if not (parent_state.keys() == random_state.keys() == child_state.keys()):
        raise AssertionError("Controller state keys differ during inheritance audit")
    copied = []
    random_kept = []
    for name, child_value in child_state.items():
        if is_combined_qkv(name):
            embed_dim = child_value.shape[0] // 3
            if not torch.equal(
                child_value[: 2 * embed_dim],
                random_state[name][: 2 * embed_dim],
            ):
                raise AssertionError(name + " did not retain random Q/K slices")
            if not torch.equal(
                child_value[2 * embed_dim:],
                parent_state[name][2 * embed_dim:],
            ):
                raise AssertionError(name + " did not copy the parent V slice")
            random_kept.append(name + "[QK]")
            copied.append(name + "[V]")
        elif _is_separate_qk_parameter(name) or _is_layernorm_parameter(name):
            if not torch.equal(child_value, random_state[name]):
                raise AssertionError(name + " did not retain random initialization")
            random_kept.append(name)
        else:
            if not torch.equal(child_value, parent_state[name]):
                raise AssertionError(name + " was not copied from the parent")
            copied.append(name)
    return {
        "copied_parameter_entries": copied,
        "random_parameter_entries": random_kept,
    }


def changed_state_entries(before_state, controller):
    return [
        name
        for name, value in controller.state_dict().items()
        if not torch.equal(before_state[name], value)
    ]


def configure_controller(env, child_body, parent_controller, seed,
                         eval_interval, num_evals, device):
    sample_setting = roaa.get_sample_setting(env, child_body)
    trans = transformerconfig()
    controller_type = getattr(parent_controller.ac, "controller_type", "original")
    trans.controller_type = controller_type
    if controller_type == "daab":
        trans.attention_heads = trans.daab_attention_heads
        trans.condition_decoder = True
    ppo_cfg = ppoconfig()
    ppo_cfg.controller_type = controller_type
    ppo_cfg.env_name = env
    ppo_cfg.seed = seed
    ppo_cfg.eval_interval = eval_interval
    ppo_cfg.num_evals = num_evals
    return sample_setting, ppo_cfg, trans


def prepare_arm_controller(arm, parent_controller, parent_obs_rms, parent_body,
                           child_body, env, seed, eval_interval, num_evals,
                           device):
    sample_setting, ppo_cfg, trans = configure_controller(
        env, child_body, parent_controller, seed, eval_interval, num_evals, device
    )
    parent_state = clone_state(parent_controller)

    roaa.set_all_seeds(seed)
    random_controller = roaa.new_random_controller(
        sample_setting, ppo_cfg, trans, device
    )
    random_state = clone_state(random_controller)

    if arm == "random_init":
        child = random_controller
        audit = {
            "copied_parameter_entries": [],
            "random_parameter_entries": list(random_state.keys()),
        }
    else:
        roaa.set_all_seeds(seed)
        child = build_copied_child_controller(
            parent_controller,
            sample_setting,
            ppo_cfg,
            trans,
            device,
            ARM_COPY_MODES[arm],
        )
        if ARM_COPY_MODES[arm] == "full_copy":
            audit = audit_full_copy(parent_state, child.state_dict())
        else:
            audit = audit_non_distill_copy(
                parent_state, random_state, child.state_dict()
            )

    warmup_changed = []
    rollout_count = 0
    if arm == DISTILL_ARM:
        before_warmup = clone_state(child)
        roaa.set_all_seeds(seed)
        observations = collect_parent_rollout_obs(
            parent_controller,
            parent_body,
            env,
            seed,
            device,
            ob_rms=parent_obs_rms,
        )
        rollout_count = len(observations)
        roaa.set_all_seeds(seed)
        attention_distill_warmup(
            parent_controller,
            child,
            observations,
            parent_body,
            child_body,
            device,
            batch_size=64,
            warmup_epochs=2,
            loss_type=DISTILL_LOSS,
            lambda_attention=1.0,
            lambda_feature=1.0,
        )
        warmup_changed = changed_state_entries(before_warmup, child)
        if not warmup_changed:
            raise AssertionError("Distillation warm-up changed no parameters")
        unexpected = [
            name for name in warmup_changed
            if not is_distillation_state_entry(name)
        ]
        if unexpected:
            raise AssertionError(
                "Warm-up changed non-distillation parameters: {}".format(unexpected)
            )

    audit.update(
        {
            "arm": arm,
            "copy_mode": ARM_COPY_MODES[arm],
            "distillation_enabled": arm == DISTILL_ARM,
            "distillation_loss": DISTILL_LOSS if arm == DISTILL_ARM else None,
            "warmup_epochs": 2 if arm == DISTILL_ARM else 0,
            "warmup_changed_parameter_entries": warmup_changed,
            "parent_rollout_observations": rollout_count,
        }
    )
    return child, ppo_cfg, audit


def write_json(path, value):
    with open(path, "w") as output_file:
        json.dump(value, output_file, indent=2)


def train_ablation_task(offspring_dir, arm, task_dir, env, updates,
                        eval_interval, num_evals, seed, torch_threads):
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")

    try:
        roaa.register_legacy_checkpoint_aliases()
        os.makedirs(os.path.join(task_dir, "controllers"), exist_ok=True)
        with open(os.path.join(offspring_dir, "metadata.json")) as metadata_file:
            metadata = json.load(metadata_file)
        parent_body_path = loss_runner.resolve_offspring_asset_path(
            offspring_dir, metadata.get("parent_body_path"), "parent_body.npz"
        )
        parent_controller_path = loss_runner.resolve_offspring_asset_path(
            offspring_dir,
            metadata.get("parent_controller_path"),
            "parent_controller.pt",
        )
        child_body_path = loss_runner.resolve_offspring_asset_path(
            offspring_dir, metadata.get("child_body_path"), "body.npz"
        )
        parent_body = roaa.load_parent_structure(parent_body_path)
        child_body = roaa.load_parent_structure(child_body_path)
        parent_controller, parent_obs_rms = roaa.load_parent_controller(
            parent_controller_path, device
        )

        child, ppo_cfg, audit = prepare_arm_controller(
            arm,
            parent_controller,
            parent_obs_rms,
            parent_body,
            child_body,
            env,
            seed,
            eval_interval,
            num_evals,
            device,
        )
        audit.update(
            {
                "offspring_index": metadata["offspring_index"],
                "seed": seed,
                "direct_parent_id": metadata["direct_parent_id"],
                "most_similar_parent_id": metadata["most_similar_parent_id"],
                "distance_bin": metadata.get("distance_bin"),
                "changed_voxel_count": metadata.get("changed_voxel_count"),
            }
        )
        write_json(os.path.join(task_dir, "initialization.json"), audit)

        roaa.set_all_seeds(seed)
        child.ac.device = device
        ppo = PPO(
            robot=(child_body, get_full_connectivity(child_body)),
            ppo_size=1,
            train_iters=updates,
            agent=child,
            verbose=True,
            ppo_args=ppo_cfg,
            total_step=updates,
            save_path=task_dir,
            device=device,
            mmse=False,
        )
        ppo.train("best", 0, float("-inf"))
        checkpoint_path = os.path.join(task_dir, "controllers", "best.pt")
        if not os.path.isfile(checkpoint_path):
            raise AssertionError("Task did not produce " + checkpoint_path)
        records = [
            {
                "offspring_index": metadata["offspring_index"],
                "arm": arm,
                "seed": seed,
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
                "Task curve updates differ from {}".format(expected_updates)
            )
        write_curve_csv(os.path.join(task_dir, "learning_curve.csv"), records)
        stderr_path = os.path.join(task_dir, "stderr.log")
        if os.path.exists(stderr_path):
            os.remove(stderr_path)
        return records
    except Exception:
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "stderr.log"), "w") as error_file:
            traceback.print_exc(file=error_file)
        raise


def write_curve_csv(path, records):
    with open(path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def read_curve_csv(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, newline="") as input_file:
        for row in csv.DictReader(input_file):
            records.append(
                {
                    "offspring_index": int(row["offspring_index"]),
                    "arm": row["arm"],
                    "seed": int(row["seed"]),
                    "update": int(row["update"]),
                    "env_steps": int(row["env_steps"]),
                    "return": float(row["return"]),
                    "best_return": float(row["best_return"]),
                }
            )
    return records


def curve_is_complete(path, updates, eval_interval):
    try:
        records = read_curve_csv(path)
    except (KeyError, TypeError, ValueError):
        return False
    return [record["update"] for record in records] == list(
        range(0, updates + 1, eval_interval)
    )


def plot_arm_curves(output_dir, records, metric, stem, ylabel, env):
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for arm in ARM_ORDER:
        arm_records = [record for record in records if record["arm"] == arm]
        if not arm_records:
            continue
        axis.plot(
            [record["update"] for record in arm_records],
            [record[metric] for record in arm_records],
            linewidth=1.8,
            label=ARM_LABELS[arm],
        )
    axis.set_xlabel("PPO updates")
    axis.set_ylabel(ylabel)
    axis.set_title("{} controller copy/distillation ablation".format(env))
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


def aggregate(offspring, args, status):
    arm_rank = {arm: index for index, arm in enumerate(ARM_ORDER)}
    summary_rows = []
    for offspring_index, (_, metadata) in sorted(offspring.items()):
        offspring_results = os.path.join(
            args.results_dir, "offspring_{}".format(offspring_index)
        )
        combined = []
        for arm in ARM_ORDER:
            records = read_curve_csv(
                os.path.join(offspring_results, arm, "learning_curve.csv")
            )
            combined.extend(records)
            row = {
                "offspring_index": offspring_index,
                "direct_parent_id": metadata["direct_parent_id"],
                "most_similar_parent_id": metadata["most_similar_parent_id"],
                "similarity_count": metadata["similarity_count"],
                "distance_bin": metadata.get("distance_bin", ""),
                "changed_voxel_count": metadata.get("changed_voxel_count", ""),
                "arm": arm,
                "status": status.get(
                    "offspring_{}/{}".format(offspring_index, arm), "not_run"
                ),
            }
            if records:
                row.update(
                    {
                        "status": "completed",
                        "points": len(records),
                        "initial_return": records[0]["return"],
                        "final_return": records[-1]["return"],
                        "best_return": records[-1]["best_return"],
                        "peak_return": max(record["return"] for record in records),
                    }
                )
            summary_rows.append(row)
        if combined:
            combined.sort(
                key=lambda record: (arm_rank[record["arm"]], record["update"])
            )
            os.makedirs(offspring_results, exist_ok=True)
            write_curve_csv(
                os.path.join(offspring_results, "learning_curves.csv"), combined
            )
            plot_arm_curves(
                offspring_results,
                combined,
                "return",
                "learning_curves_raw",
                "Evaluation return",
                args.env,
            )
            plot_arm_curves(
                offspring_results,
                combined,
                "best_return",
                "learning_curves_best_so_far",
                "Best-so-far evaluation return",
                args.env,
            )

    fields = (
        "offspring_index",
        "direct_parent_id",
        "most_similar_parent_id",
        "similarity_count",
        "distance_bin",
        "changed_voxel_count",
        "arm",
        "status",
        "points",
        "initial_return",
        "final_return",
        "best_return",
        "peak_return",
    )
    with open(
        os.path.join(args.results_dir, "summary.csv"), "w", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    args = parse_args()
    validate_args(args)
    os.makedirs(args.results_dir, exist_ok=True)
    offspring = loss_runner.discover_offspring(args)
    selected = sorted(offspring.keys())
    arms = ARM_ORDER if args.only_arm is None else (args.only_arm,)
    resolved_parallel, resolved_threads = loss_runner.resolve_resource_config(
        args.max_parallel, args.torch_threads
    )
    offspring_per_wave = max(1, resolved_parallel // len(arms))
    waves = [
        selected[index:index + offspring_per_wave]
        for index in range(0, len(selected), offspring_per_wave)
    ]

    config = vars(args).copy()
    config.update(
        {
            "arm_order": list(arms),
            "all_arm_definitions": dict(ARM_COPY_MODES),
            "distillation_arm": DISTILL_ARM,
            "distillation_loss": DISTILL_LOSS,
            "warmup_epochs": 2,
            "lambda_attention": 1.0,
            "lambda_feature": 1.0,
            "selected_offspring": selected,
            "logical_cpus": os.cpu_count() or 1,
            "resolved_max_parallel": resolved_parallel,
            "resolved_torch_threads": resolved_threads,
            "offspring_per_wave": offspring_per_wave,
        }
    )
    write_json(os.path.join(args.results_dir, "config.json"), config)

    status_path = os.path.join(args.results_dir, "tasks_status.json")
    status = {}
    if os.path.exists(status_path) and not args.force:
        with open(status_path) as status_file:
            status = json.load(status_file)

    wave_tasks = []
    for wave in waves:
        tasks = []
        for offspring_index in wave:
            offspring_dir, _ = offspring[offspring_index]
            seed = args.seed_base + offspring_index
            for arm in arms:
                key = "offspring_{}/{}".format(offspring_index, arm)
                task_dir = os.path.join(
                    args.results_dir,
                    "offspring_{}".format(offspring_index),
                    arm,
                )
                curve_path = os.path.join(task_dir, "learning_curve.csv")
                if curve_is_complete(
                    curve_path, args.updates, args.eval_interval
                ) and not args.force:
                    status[key] = "completed"
                    continue
                status[key] = "pending"
                tasks.append(
                    (
                        offspring_dir,
                        arm,
                        task_dir,
                        args.env,
                        args.updates,
                        args.eval_interval,
                        args.num_evals,
                        seed,
                        resolved_threads,
                    )
                )
        wave_tasks.append(tasks)
    write_json(status_path, status)

    total_tasks = sum(len(tasks) for tasks in wave_tasks)
    if total_tasks == 0:
        print("No pending tasks; aggregating existing results only.")
    else:
        print(
            "Running {} tasks in {} waves with up to {} processes and {} "
            "Torch thread(s) each.".format(
                total_tasks, len(waves), resolved_parallel, resolved_threads
            ),
            flush=True,
        )
        spawn_context = multiprocessing.get_context("spawn")
        for wave_index, tasks in enumerate(wave_tasks):
            if not tasks:
                continue
            print(
                "\n=== Wave {}: {} tasks ===".format(wave_index + 1, len(tasks)),
                flush=True,
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(resolved_parallel, len(tasks)),
                mp_context=spawn_context,
            ) as executor:
                futures = {
                    executor.submit(train_ablation_task, *task): task
                    for task in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    offspring_index = int(
                        os.path.basename(task[0]).split("_")[-1]
                    )
                    arm = task[1]
                    key = "offspring_{}/{}".format(offspring_index, arm)
                    try:
                        future.result()
                        status[key] = "completed"
                    except Exception:
                        status[key] = "failed"
                    write_json(status_path, status)
                    print(
                        "Finished offspring_{} / {} [{}]".format(
                            offspring_index, arm, status[key]
                        ),
                        flush=True,
                    )

    aggregate(offspring, args, status)
    print("All waves done. Results under " + os.path.abspath(args.results_dir))


if __name__ == "__main__":
    main()
