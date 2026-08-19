"""Batch runner for the attention-loss ablation: 5 offspring x 4 loss values.

Each task inherits the most similar parent's controller, warms it up with one
of the four attention_distill_loss values, then trains PPO for `updates`
iterations and records a learning curve. Tasks run in spawn subprocesses with
at most `--max-parallel` concurrent workers, submitted in strict offspring
waves (2 offspring, then 2, then 1), so iteration 1 uses 8, then 8, then 4.
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
        description="Run the 5-offspring x 4-loss attention distillation ablation."
    )
    parser.add_argument(
        "--offspring-dir",
        default=os.path.join(ROOT_DIR, "result", "attention_loss_ablation", "offspring"),
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(ROOT_DIR, "result", "attention_loss_ablation", "results"),
    )
    parser.add_argument("--env", default="Walker-v0")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--num-evals", type=int, default=2)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=2000,
        help="Offspring i uses training seed seed_base + i for all 4 losses.",
    )
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
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
    if not 1 <= args.max_parallel <= 8:
        raise ValueError("--max-parallel must be between 1 and 8")
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")


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


def train_ablation_task(
    offspring_dir, loss_name, task_dir, env, updates, eval_interval,
    num_evals, seed, torch_threads,
):
    """Spawn-process worker: load parent, inherit, warm up with one loss, train PPO.

    Everything is built inside the worker so no PPOAgent is ever pickled.
    """
    torch.set_num_threads(torch_threads)
    device = torch.device("cpu")
    try:
        roaa.register_legacy_checkpoint_aliases()
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(os.path.join(task_dir, "controllers"), exist_ok=True)
        with open(os.path.join(offspring_dir, "metadata.json")) as f:
            meta = json.load(f)
        parent_body = roaa.load_parent_structure(meta["parent_body_path"])
        parent_ctrl, parent_obs_rms = roaa.load_parent_controller(
            meta["parent_controller_path"], device
        )
        child_body = roaa.load_parent_structure(meta["child_body_path"])

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
                "similarity_count", "loss", "status", "points", "initial_return",
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
    waves = [selected[i:i + 2] for i in range(0, len(selected), 2)]
    losses = LOSS_ORDER if args.only_loss is None else (args.only_loss,)

    config = vars(args).copy()
    config["loss_order"] = list(losses)
    config["selected_offspring"] = selected
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
                if os.path.exists(curve_path) and not args.force:
                    status[key] = status.get(key, "skipped")
                    continue
                status[key] = "pending"
                tasks.append(
                    (
                        off_dir, loss, task_dir, args.env, args.updates,
                        args.eval_interval, args.num_evals, seed, args.torch_threads,
                    )
                )
        wave_task_lists.append(tasks)

    total_tasks = sum(len(tasks) for tasks in wave_task_lists)
    if total_tasks == 0:
        print("No pending tasks; aggregating existing results only.")
    else:
        max_workers = min(
            args.max_parallel, max(len(tasks) for tasks in wave_task_lists)
        )
        print(
            "\nRunning {} tasks in {} waves with up to {} concurrent processes...".format(
                total_tasks, len(waves), max_workers
            )
        )
        spawn_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=spawn_ctx
        ) as executor:
            for wave_index, tasks in enumerate(wave_task_lists):
                if not tasks:
                    continue
                print(
                    "\n=== Wave {}: {} tasks ===".format(wave_index + 1, len(tasks)),
                    flush=True,
                )
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