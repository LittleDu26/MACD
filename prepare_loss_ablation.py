"""Select the best MACD(2loss) parents, mutate five offspring with MACD's own
mutation operator, and persist a reusable offspring folder for the
attention-loss ablation batch runner (run_loss_ablation_tasks.py).

Offspring generation follows the MACD flow (utils.algo_utils.mutate), and each
offspring is paired with its most similar parent according to MACD's
same-nonempty-voxel similarity (macd.controller_distillation.same_voxel_mask).
"""

import argparse
import csv
import json
import os
import random
import re
import shutil

import numpy as np
from evogym import get_full_connectivity, has_actuator, hashable, is_connected

import run_offspring_attention_ablation as roaa
from macd.controller_distillation import same_voxel_mask, should_skip_warmup
from utils.algo_utils import mutate

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

GEN_RE = re.compile(r"Running generation (\d+)")
SURV_RE = re.compile(r"survivor IDs\(\d+\):\[([^\]]+)\]")
ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s*$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select MACD parents and generate offspring for the loss ablation."
    )
    parser.add_argument(
        "--result-dir",
        default=os.path.join(ROOT_DIR, "result", "MACD(2loss)", "Walker-v0", "0"),
        help="Existing MACD run directory containing out.txt, structures/ and controllers/.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT_DIR, "result", "attention_loss_ablation"),
        help="Output folder receiving parents/ and offspring/.",
    )
    parser.add_argument("--env", default="Walker-v0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-k", type=int, default=5)
    parser.add_argument("--sampling-seed", type=int, default=20260818)
    parser.add_argument("--mutation-seed-base", type=int, default=1000)
    parser.add_argument("--max-mutation-retries", type=int, default=100)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-parse, re-sample and re-mutate even if metadata.json exists.",
    )
    return parser.parse_args()


def parse_last_completed_generation(out_path):
    """Return the last generation block of out.txt that recorded a promotion.

    Each completed generation ends with a line like
    ``survivor IDs(10):[1076, 1024, ...]``; the (interrupted) final generation
    has none. Blocks are delimited by ``Running generation N`` headers.
    """
    with open(out_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    blocks = []
    starts = [i for i, line in enumerate(lines) if GEN_RE.search(line)]
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        block = lines[start:end]
        match = next(
            (SURV_RE.search(line) for line in block if SURV_RE.search(line)),
            None,
        )
        if match is None:
            continue  # interrupted generation, no promotion recorded
        generation = int(GEN_RE.search(lines[start]).group(1))
        survivor_ids = [int(x) for x in match.group(1).split(",")]
        table = {}
        in_table = False
        for line in block:
            if "fitness" in line:
                in_table = True
                continue
            if "Start" in line:
                in_table = False
                continue
            if in_table:
                m = ROW_RE.match(line)
                if m:
                    table[int(m.group(1))] = (int(m.group(2)), float(m.group(3)))
        blocks.append(
            {"generation": generation, "survivor_ids": survivor_ids, "table": table}
        )
    if not blocks:
        raise RuntimeError("No completed generation block found in " + out_path)
    return blocks[-1]


def make_offspring(parent_body, seed, record, max_retries):
    """Mutate one parent with MACD's exact operator until a valid child appears."""
    roaa.set_all_seeds(seed)
    for attempt in range(1, max_retries + 1):
        child = mutate(parent_body, record)
        if child is None:
            continue
        if np.array_equal(child, parent_body):
            continue
        if not is_connected(child) or not has_actuator(child):
            continue
        if should_skip_warmup(parent_body, child):
            continue
        return np.asarray(child).copy(), attempt
    raise RuntimeError(
        "No valid offspring after {} mutate() calls".format(max_retries)
    )


def most_similar_parent(child_body, top_ids, bodies):
    """MACD similarity: count of equal nonempty voxels, ties favor earlier ids."""
    best_id, best_count = None, -1
    for pid in top_ids:
        count = int(np.sum(same_voxel_mask(bodies[pid], child_body)))
        if count > best_count:
            best_id, best_count = pid, count
    return best_id, best_count


def print_summary(top_meta):
    print("=== Parents (generation {}) ===".format(top_meta["generation"]))
    for parent in top_meta["parents"]:
        print(
            "  id {:<6} maturity {:<3} fitness {:.6f}".format(
                parent["id"], parent["maturity"], parent["fitness"]
            )
        )
    print(
        "=== Sampled parents (seed {}) ===\n  {}".format(
            top_meta["sampling_seed"], top_meta["sampled_parent_ids"]
        )
    )
    offspring_meta_paths = sorted(
        os.path.join(top_meta["offspring_dir"], name, "metadata.json")
        for name in os.listdir(top_meta["offspring_dir"])
        if name.startswith("offspring_")
        and os.path.isdir(os.path.join(top_meta["offspring_dir"], name))
    )
    print("=== Offspring -> most similar parent ===")
    for path in offspring_meta_paths:
        with open(path) as f:
            meta = json.load(f)
        print(
            "  offspring_{}: direct parent {} -> most similar parent {} "
            "(similarity {}), changed voxels {}, attempt {}".format(
                meta["offspring_index"],
                meta["direct_parent_id"],
                meta["most_similar_parent_id"],
                meta["similarity_count"],
                meta["changed_voxel_count"],
                meta["mutation_attempt"],
            )
        )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    top_meta_path = os.path.join(args.output_dir, "metadata.json")

    if os.path.exists(top_meta_path) and not args.regenerate:
        print("Reusing existing experiment metadata: " + top_meta_path)
        with open(top_meta_path) as f:
            print_summary(json.load(f))
        return

    out_path = os.path.join(args.result_dir, "out.txt")
    gen_info = parse_last_completed_generation(out_path)
    survivor_ids = gen_info["survivor_ids"]
    table = gen_info["table"]
    if len(survivor_ids) < args.top_k:
        raise RuntimeError(
            "Last completed generation has {} survivors, fewer than top-k {}".format(
                len(survivor_ids), args.top_k
            )
        )
    top_ids = survivor_ids[: args.top_k]
    table_sorted = [pid for pid, _ in sorted(table.items(), key=lambda kv: -kv[1][1])]
    if table_sorted[: args.top_k] != top_ids:
        print(
            "WARNING: survivor order differs from the printed fitness order; "
            "proceeding with survivor order"
        )

    bodies = {}
    for pid in top_ids:
        structure_path = os.path.join(args.result_dir, "structures", "{}.npz".format(pid))
        controller_path = os.path.join(args.result_dir, "controllers", "{}.pt".format(pid))
        if not os.path.exists(structure_path) or not os.path.exists(controller_path):
            raise RuntimeError(
                "Missing structure/controller checkpoint for parent {} in {}".format(
                    pid, args.result_dir
                )
            )
        bodies[pid] = roaa.load_parent_structure(structure_path)

    parents_dir = os.path.join(args.output_dir, "parents")
    offspring_dir = os.path.join(args.output_dir, "offspring")
    os.makedirs(parents_dir, exist_ok=True)
    os.makedirs(offspring_dir, exist_ok=True)
    for pid in top_ids:
        shutil.copy2(
            os.path.join(args.result_dir, "structures", "{}.npz".format(pid)),
            os.path.join(parents_dir, "{}.npz".format(pid)),
        )
        shutil.copy2(
            os.path.join(args.result_dir, "controllers", "{}.pt".format(pid)),
            os.path.join(parents_dir, "{}.pt".format(pid)),
        )
    with open(os.path.join(args.output_dir, "parents.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("id", "maturity", "fitness"))
        writer.writeheader()
        for pid in top_ids:
            maturity, fitness = table[pid]
            writer.writerow({"id": pid, "maturity": maturity, "fitness": fitness})

    roaa.set_all_seeds(args.sampling_seed)
    sampled = random.sample(top_ids, args.sample_k)
    record = {hashable(body): [] for body in bodies.values()}

    for i, pid in enumerate(sampled):
        parent_body = bodies[pid]
        seed = args.mutation_seed_base + i
        child_body, attempt = make_offspring(
            parent_body, seed, record, args.max_mutation_retries
        )
        msp_id, sim_count = most_similar_parent(child_body, top_ids, bodies)
        morph = roaa.morphology_metadata(parent_body, child_body, attempt)

        child_dir = os.path.join(offspring_dir, "offspring_{}".format(i))
        os.makedirs(child_dir, exist_ok=True)
        child_npz_path = os.path.join(child_dir, "body.npz")
        parent_npz_path = os.path.join(child_dir, "parent_body.npz")
        controller_path = os.path.join(child_dir, "parent_controller.pt")
        np.savez(child_npz_path, child_body, get_full_connectivity(child_body))
        np.savez(parent_npz_path, bodies[msp_id], get_full_connectivity(bodies[msp_id]))
        shutil.copy2(
            os.path.join(parents_dir, "{}.pt".format(msp_id)), controller_path
        )

        meta = {
            "offspring_index": i,
            "direct_parent_id": pid,
            "most_similar_parent_id": msp_id,
            "similarity_count": sim_count,
            "shared_nonempty_voxel_count": morph["shared_nonempty_voxel_count"],
            "mutation_seed": seed,
            "mutation_attempt": attempt,
            "changed_voxel_count": morph["changed_voxel_count"],
            "changed_voxels": morph["changed_voxels"],
            "parent_body": morph["parent_body"],
            "child_body": morph["child_body"],
            "parent_body_hash": hashable(bodies[msp_id]),
            "child_body_hash": hashable(child_body),
            "parent_body_path": os.path.abspath(parent_npz_path),
            "parent_controller_path": os.path.abspath(controller_path),
            "child_body_path": os.path.abspath(child_npz_path),
        }
        with open(os.path.join(child_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    top_meta = {
        "result_dir": os.path.abspath(args.result_dir),
        "env": args.env,
        "top_k": args.top_k,
        "sample_k": args.sample_k,
        "sampling_seed": args.sampling_seed,
        "mutation_seed_base": args.mutation_seed_base,
        "max_mutation_retries": args.max_mutation_retries,
        "generation": gen_info["generation"],
        "offspring_dir": os.path.abspath(offspring_dir),
        "parents": [
            {
                "id": pid,
                "maturity": table[pid][0],
                "fitness": table[pid][1],
                "structure_path": os.path.abspath(
                    os.path.join(parents_dir, "{}.npz".format(pid))
                ),
                "controller_path": os.path.abspath(
                    os.path.join(parents_dir, "{}.pt".format(pid))
                ),
            }
            for pid in top_ids
        ],
        "sampled_parent_ids": sampled,
    }
    with open(top_meta_path, "w") as f:
        json.dump(top_meta, f, indent=2)

    print_summary(top_meta)
    print("Experiment prepared under " + os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()