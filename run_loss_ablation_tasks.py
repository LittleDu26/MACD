"""Batch runner for a morphology-stratified attention-loss ablation.

Each task inherits the most similar parent's controller, warms it up with one
of the four attention_distill_loss values, then trains PPO for `updates`
iterations and records a learning curve. Tasks run in spawn subprocesses and
are grouped into offspring waves sized from the resolved process count.
"""

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import os
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from evogym import get_full_connectivity

import run_offspring_attention_ablation as roaa
from macd.controller_distillation import (
    DISTILL_LOSS_TYPES,
    attention_distill_warmup,
    build_child_controller,
    collect_parent_rollout_obs,
)
from macd.ppo import PPO
from macd.transformer.config import ppoconfig, transformerconfig

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_PARALLEL_LIMIT = 20
LOSS_ORDER = tuple(DISTILL_LOSS_TYPES)
LOSS_LABELS = {
    "column_kl": "Column KL",
    "shared_attention": "Shared attention KL",
    "shared_feature": "Shared feature MSE",
    "shared_attention_feature": "Shared attention + feature",
}
CURVE_FIELDS = (
    "offspring_index",
    "loss",
    "seed",
    "update",
    "env_steps",
    "return",
    "best_return",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the stratified attention-loss distillation ablation."
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
            "attention_loss_ablation_stratified_20",
            "results",
        ),
    )
    parser.add_argument("--env", default="Walker-v0")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--num-evals", type=int, default=2)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=2000,
        help="Offspring i uses training seed seed_base + i for all 4 losses.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Concurrent worker processes; 0 auto-detects (maximum 20).",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Torch/OMP threads per worker; 0 auto-detects.",
    )
    parser.add_argument(
        "--offspring-ids",
        type=int,
        nargs="+",
        default=None,
        help="Restrict to these offspring indexes (smoke/recovery runs).",
    )
    parser.add_argument(
        "--only-loss",
        choices=LOSS_ORDER,
        default=None,
        help="Restrict to one loss value (smoke/recovery runs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run tasks whose learning_curve.csv already exists.",
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
    if not 0 <= args.max_parallel <= MAX_PARALLEL_LIMIT:
        raise ValueError(
            "--max-parallel must be between 0 and {}".format(MAX_PARALLEL_LIMIT)
        )
    if args.torch_threads < 0:
        raise ValueError("--torch-threads must be nonnegative")


def resolve_resource_config(max_parallel, torch_threads, logical_cpus=None):
    """Resolve process and per-process thread counts, reserving two CPUs."""
    logical_cpus = logical_cpus or os.cpu_count() or 1
    available_cpus = max(1, logical_cpus - 2)
    resolved_parallel = (
        min(MAX_PARALLEL_LIMIT, available_cpus)
        if max_parallel == 0
        else max_parallel
    )
    resolved_threads = (
        max(1, available_cpus // resolved_parallel)
        if torch_threads == 0
        else torch_threads
    )
    return resolved_parallel, resolved_threads


def discover_offspring(args):
    os.makedirs(args.offspring_dir, exist_ok=True)
    offspring = {}
    for name in sorted(os.listdir(args.offspring_dir)):
        dir_path = os.path.join(args.offspring_dir, name)
        if not name.startswith("offspring_") or not os.path.isdir(dir_path):
            continue
        meta_path = os.path.join(dir_path, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        offspring[meta["offspring_index"]] = (dir_path, meta)
    if args.offspring_ids is not None:
        offspring = {
            idx: value
            for idx, value in offspring.items()
            if idx in args.offspring_ids
        }
    if not offspring:
        raise RuntimeError("No offspring metadata found under " + args.offspring_dir)
    return offspring


def resolve_offspring_asset_path(offspring_dir, configured_path, filename):
    """Resolve metadata paths after an experiment directory is relocated."""
    candidates = []
    if configured_path:
        candidates.append(configured_path)
        if not os.path.isabs(configured_path):
            candidates.append(os.path.join(offspring_dir, configured_path))
    candidates.append(os.path.join(offspring_dir, filename))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not resolve {} from offspring directory {}".format(
            filename, offspring_dir
        )
    )


def train_ablation_task(
    offspring_dir, loss_name, task_dir, env, updates, eval_interval,
    num_evals, seed, torch_threads,
):
    """Spawn-process worker: load parent, inherit, warm up with one loss, train PPO.

    Everything is built inside the worker so no PPOAgent is ever pickled.
    """
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
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(os.path.join(task_dir, "controllers"), exist_ok=True)
        with open(os.path.join(offspring_dir, "metadata.json")) as f:
            meta = json.load(f)
        parent_body_path = resolve_offspring_asset_path(
            offspring_dir, meta.get("parent_body_path"), "parent_body.npz"
        )
        parent_controller_path = resolve_offspring_asset_path(
            offspring_dir, meta.get("parent_controller_path"), "parent_controller.pt"
        )
        child_body_path = resolve_offspring_asset_path(
            offspring_dir, meta.get("child_body_path"), "body.npz"
        )
        parent_body = roaa.load_parent_structure(parent_body_path)
        parent_ctrl, parent_obs_rms = roaa.load_parent_controller(
            parent_controller_path, device
        )
        child_body = roaa.load_parent_structure(child_body_path)

        sample_setting = roaa.get_sample_setting(env, child_body)
        trans = transformerconfig()
        ctype = getattr(parent_ctrl.ac, "controller_type", "original")
        trans.controller_type = ctype
        if ctype == "daab":
            trans.attention_heads = trans.daab_attention_heads
            trans.condition_decoder = True
        ppo_cfg = ppoconfig()
        ppo_cfg.controller_type = ctype
        ppo_cfg.env_name = env
        ppo_cfg.seed = seed
        ppo_cfg.eval_interval = eval_interval
        ppo_cfg.num_evals = num_evals

        roaa.set_all_seeds(seed)
        child = build_child_controller(parent_ctrl, sample_setting, ppo_cfg, trans, device)
        roaa.set_all_seeds(seed)
        observations = collect_parent_rollout_obs(
            parent_ctrl, parent_body, env, seed, device, ob_rms=parent_obs_rms
        )
        roaa.set_all_seeds(seed)
        attention_distill_warmup(
            parent_ctrl,
            child,
            observations,
            parent_body,
            child_body,
            device,
            loss_type=loss_name,
            batch_size=64,
            warmup_epochs=2,
            lambda_attention=1.0,
            lambda_feature=1.0,
        )

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

        records = [
            {
                "offspring_index": meta["offspring_index"],
                "loss": loss_name,
                "seed": seed,
                "update": record["update"],
                "env_steps": record["env_steps"],
                "return": record["return"],
                "best_return": record["best_return"],
            }
            for record in ppo.evaluation_records
        ]
        expected = 1 + updates // eval_interval
        if len(records) != expected or records[-1]["update"] != updates:
            raise AssertionError(
                "Task produced {} curve points; expected {} ending at update {}".format(
                    len(records), expected, updates
                )
            )
        write_curve_csv(os.path.join(task_dir, "learning_curve.csv"), records)
        stderr_path = os.path.join(task_dir, "stderr.log")
        if os.path.exists(stderr_path):
            os.remove(stderr_path)
        return records
    except Exception:
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "stderr.log"), "w") as f:
            traceback.print_exc(file=f)
        raise


def write_curve_csv(path, records):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def read_curve_csv(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            records.append(
                {
                    "offspring_index": int(row["offspring_index"]),
                    "loss": row["loss"],
                    "seed": int(row["seed"]),
                    "update": int(row["update"]),
                    "env_steps": int(row["env_steps"]),
                    "return": float(row["return"]),
                    "best_return": float(row["best_return"]),
                }
            )
    return records


def curve_is_complete(path, updates, eval_interval):
    expected = 1 + updates // eval_interval
    try:
        records = read_curve_csv(path)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        len(records) == expected
        and records[0]["update"] == 0
        and records[-1]["update"] == updates
    )


def plot_loss_curves(output_dir, records, metric, stem, ylabel, env_name):
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for loss in LOSS_ORDER:
        loss_records = [r for r in records if r["loss"] == loss]
        if not loss_records:
            continue
        axis.plot(
            [r["update"] for r in loss_records],
            [r[metric] for r in loss_records],
            linewidth=1.8,
            label=LOSS_LABELS[loss],
        )
    axis.set_xlabel("PPO updates")
    axis.set_ylabel(ylabel)
    axis.set_title("{} offspring loss ablation".format(env_name))
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


def aggregate(offspring, args):
    """Write per-offspring combined CSVs/plots and the global summary."""
    loss_rank = {loss: i for i, loss in enumerate(LOSS_ORDER)}
    summary_rows = []
    for idx, (_, meta) in sorted(offspring.items()):
        combined = []
        off_dir = os.path.join(args.results_dir, "offspring_{}".format(idx))
        for loss in LOSS_ORDER:
            records = read_curve_csv(
                os.path.join(off_dir, loss, "learning_curve.csv")
            )
            combined.extend(records)
            rows = {"offspring_index": idx, "loss": loss}
            rows.update(
                {
                    "direct_parent_id": meta["direct_parent_id"],
                    "most_similar_parent_id": meta["most_similar_parent_id"],
                    "similarity_count": meta["similarity_count"],
                    "distance_bin": meta.get("distance_bin", ""),
                    "changed_voxel_count": meta.get("changed_voxel_count", ""),
                }
            )
            if records:
                rows.update(
                    {
                        "status": "completed",
                        "points": len(records),
                        "initial_return": records[0]["return"],
                        "final_return": records[-1]["return"],
                        "best_return": records[-1]["best_return"],
                        "peak_return": max(r["return"] for r in records),
                    }
                )
            summary_rows.append(rows)
        if combined:
            combined.sort(key=lambda r: (loss_rank[r["loss"]], r["update"]))
            write_curve_csv(os.path.join(off_dir, "learning_curves.csv"), combined)
            plot_loss_curves(
                off_dir, combined, "return", "learning_curves_raw",
                "Evaluation return", args.env,
            )
            plot_loss_curves(
                off_dir, combined, "best_return", "learning_curves_best_so_far",
                "Best-so-far evaluation return", args.env,
            )

    status = {}
    status_path = os.path.join(args.results_dir, "tasks_status.json")
    if os.path.exists(status_path):
        with open(status_path) as f:
            status = json.load(f)

    loss_set = LOSS_ORDER if args.only_loss is None else (args.only_loss,)
    with open(os.path.join(args.results_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "offspring_index", "direct_parent_id", "most_similar_parent_id",
                "similarity_count", "distance_bin", "changed_voxel_count",
                "loss", "status", "points", "initial_return",
                "final_return", "best_return", "peak_return",
            ),
        )
        writer.writeheader()
        for row in summary_rows:
            key = "offspring_{}/{}".format(row["offspring_index"], row["loss"])
            if row["loss"] not in loss_set:
                continue
            if "status" not in row:
                row["status"] = status.get(key, "not_run")
            writer.writerow(row)


def main():
    args = parse_args()
    validate_args(args)
    os.makedirs(args.results_dir, exist_ok=True)

    offspring = discover_offspring(args)
    selected = sorted(offspring.keys())
    losses = LOSS_ORDER if args.only_loss is None else (args.only_loss,)
    resolved_parallel, resolved_threads = resolve_resource_config(
        args.max_parallel, args.torch_threads
    )
    offspring_per_wave = max(1, resolved_parallel // len(losses))
    waves = [
        selected[index:index + offspring_per_wave]
        for index in range(0, len(selected), offspring_per_wave)
    ]

    config = vars(args).copy()
    config["loss_order"] = list(losses)
    config["selected_offspring"] = selected
    config["logical_cpus"] = os.cpu_count() or 1
    config["resolved_max_parallel"] = resolved_parallel
    config["resolved_torch_threads"] = resolved_threads
    config["offspring_per_wave"] = offspring_per_wave
    with open(os.path.join(args.results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    status = {}
    status_path = os.path.join(args.results_dir, "tasks_status.json")
    if os.path.exists(status_path) and not args.force:
        with open(status_path) as f:
            status = json.load(f)

    wave_task_lists = []
    for wave in waves:
        tasks = []
        for idx in wave:
            off_dir, meta = offspring[idx]
            seed = args.seed_base + idx
            for loss in losses:
                key = "offspring_{}/{}".format(idx, loss)
                task_dir = os.path.join(
                    args.results_dir, "offspring_{}".format(idx), loss
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
                        off_dir, loss, task_dir, args.env, args.updates,
                        args.eval_interval, args.num_evals, seed, resolved_threads,
                    )
                )
        wave_task_lists.append(tasks)

    total_tasks = sum(len(tasks) for tasks in wave_task_lists)
    if total_tasks == 0:
        print("No pending tasks; aggregating existing results only.")
    else:
        max_workers = min(
            resolved_parallel, max(len(tasks) for tasks in wave_task_lists)
        )
        print(
            "\nRunning {} tasks in {} waves with up to {} concurrent processes "
            "and {} Torch thread(s) each...".format(
                total_tasks, len(waves), max_workers, resolved_threads
            )
        )
        spawn_ctx = multiprocessing.get_context("spawn")
        for wave_index, tasks in enumerate(wave_task_lists):
            if not tasks:
                continue
            print(
                "\n=== Wave {}: {} tasks ===".format(wave_index + 1, len(tasks)),
                flush=True,
            )
            # EvoGym's native simulator does not reliably survive reuse across
            # multiple large task waves. A fresh pool also releases model and
            # simulator memory deterministically between offspring pairs.
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(resolved_parallel, len(tasks)),
                mp_context=spawn_ctx,
            ) as executor:
                futures = {
                    executor.submit(train_ablation_task, *task): task for task in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    idx = int(os.path.basename(task[0]).split("_")[-1])
                    loss = task[1]
                    key = "offspring_{}/{}".format(idx, loss)
                    try:
                        future.result()
                        status[key] = "completed"
                    except Exception:
                        status[key] = "failed"
                    print(
                        "Finished task offspring_{} / {} [{}]".format(
                            idx, loss, status[key]
                        ),
                        flush=True,
                    )
                    with open(status_path, "w") as f:
                        json.dump(status, f, indent=2)

    aggregate(offspring, args)
    print("\nAll waves done. Results under " + os.path.abspath(args.results_dir))


if __name__ == "__main__":
    main()
