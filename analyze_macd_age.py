"""Analyze MACD_age runs for Walker-v0 and Thrower-v0.

The primary learning curves use the best-so-far fitness after each logged
generation.  The summary curves use the maximum fitness observed in each
total_maturity run, followed by the cumulative maximum in the ordered
total_maturity sweep.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("result/MACD_age")
OUT = ROOT / "analysis"
TOTAL_MATURITIES = [5, 10, 15, 20, 25, 50]
ENVS = ["Walker-v0", "Thrower-v0"]

# High-contrast, color-blind-friendly categorical palette.  The previous
# plot used six shades of cyan, which made overlapping maturity curves hard to
# distinguish.
MATURITY_COLORS = {
    5: "#0072B2",   # blue
    10: "#E69F00",  # orange
    15: "#009E73",  # green
    20: "#D55E00",  # vermillion
    25: "#CC79A7",  # purple/pink
    50: "#56B4E9",  # sky blue
}
MATURITY_LINESTYLES = {
    5: "-",
    10: "--",
    15: "-.",
    20: ":",
    25: (0, (5, 1)),
    50: (0, (3, 1, 1, 1)),
}
MATURITY_MARKERS = {5: "o", 10: "s", 15: "^", 20: "D", 25: "P", 50: "X"}

GEN_RE = re.compile(r"Running generation\s+(\d+)")
BEST_RE = re.compile(
    r"Population of\s+\d+ members\s+best_id:\(ID\d+,([-+0-9.eE]+)\)"
)


def parse_logged_generation_bests(path: Path) -> list[tuple[int, float]]:
    """Return (generation, population-best) pairs from out.txt."""
    current_generation = None
    records: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        generation_match = GEN_RE.search(line)
        if generation_match:
            current_generation = int(generation_match.group(1))
            continue
        best_match = BEST_RE.search(line)
        if best_match and current_generation is not None:
            records.append((current_generation, float(best_match.group(1))))

    # There should be exactly one population-best line per generation.  Keep
    # the last line if a log was duplicated or appended during a restart.
    by_generation = {generation: value for generation, value in records}
    return sorted(by_generation.items())


def collect_run(env: str, total_maturity: int) -> dict:
    run_dir = ROOT / f"maturity_{total_maturity}" / env / "0"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    logged = parse_logged_generation_bests(run_dir / "out.txt")
    if not logged:
        raise RuntimeError(f"No generation records found in {run_dir / 'out.txt'}")

    generation = np.array([item[0] for item in logged], dtype=int)
    population_best = np.array([item[1] for item in logged], dtype=float)
    best_so_far = np.maximum.accumulate(population_best)
    # current_iters is incremented once per active agent before a generation;
    # pop_size * train_iters is therefore the comparable budget per generation.
    budget = (generation + 1) * int(config["pop_size"]) * int(config["train_iters"])

    complete = (run_dir / "historical_archive.json").exists() and (
        run_dir / "all_agents.json"
    ).exists()
    return {
        "env": env,
        "total_maturity": total_maturity,
        "run_dir": str(run_dir),
        "train_iters": int(config["train_iters"]),
        "max_iters": int(config["max_iters"]),
        "last_generation": int(generation[-1]),
        "last_budget": int(budget[-1]),
        "complete": complete,
        "generation": generation,
        "budget": budget,
        "population_best": population_best,
        "best_so_far": best_so_far,
        "run_max_fit": float(best_so_far[-1]),
    }


def write_curve_csv(runs: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "env",
                "total_maturity",
                "generation",
                "ppo_updates",
                "population_best_fit",
                "best_so_far_fit",
                "run_complete",
            ]
        )
        for run in runs:
            for generation, budget, population_best, best_so_far in zip(
                run["generation"],
                run["budget"],
                run["population_best"],
                run["best_so_far"],
            ):
                writer.writerow(
                    [
                        run["env"],
                        run["total_maturity"],
                        int(generation),
                        int(budget),
                        f"{population_best:.12g}",
                        f"{best_so_far:.12g}",
                        run["complete"],
                    ]
                )


def write_summary_csv(runs: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "env",
                "total_maturity",
                "run_max_fit",
                "cumulative_best_fit",
                "last_generation",
                "last_ppo_updates",
                "configured_max_ppo_updates",
                "run_complete",
            ]
        )
        for env in ENVS:
            env_runs = sorted(
                [run for run in runs if run["env"] == env],
                key=lambda run: run["total_maturity"],
            )
            cumulative = -np.inf
            for run in env_runs:
                cumulative = max(cumulative, run["run_max_fit"])
                writer.writerow(
                    [
                        env,
                        run["total_maturity"],
                        f"{run['run_max_fit']:.12g}",
                        f"{cumulative:.12g}",
                        run["last_generation"],
                        run["last_budget"],
                        run["max_iters"],
                        run["complete"],
                    ]
                )


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, color="#D9DEE7", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")
    ax.tick_params(colors="#374151")


def plot_learning_curves(env: str, runs: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.5), dpi=150)
    for run in sorted(runs, key=lambda item: item["total_maturity"]):
        total_maturity = run["total_maturity"]
        label = f"total_maturity={run['total_maturity']}"
        if not run["complete"]:
            label += " (incomplete)"
        ax.plot(
            run["budget"] / 1000.0,
            run["best_so_far"],
            color=MATURITY_COLORS[total_maturity],
            linewidth=2.4,
            linestyle=MATURITY_LINESTYLES[total_maturity],
            marker=MATURITY_MARKERS[total_maturity],
            markersize=3.0,
            markevery=max(1, len(run["budget"]) // 18),
            label=label,
        )
    style_axes(ax)
    ax.set_title(f"{env}: best-so-far fitness across maturity settings", loc="left", pad=12)
    ax.set_xlabel("Cumulative PPO training updates (×10³)")
    ax.set_ylabel("Best-so-far fitness")
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=9)
    ax.text(
        0,
        -0.19,
        "Each point is the maximum fitness observed up to that budget. "
        "The incomplete run is marked in the legend.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_maturity_summary(env: str, runs: list[dict], path: Path) -> None:
    ordered = sorted(runs, key=lambda item: item["total_maturity"])
    x = np.array([run["total_maturity"] for run in ordered])
    run_max = np.array([run["run_max_fit"] for run in ordered])
    cumulative = np.maximum.accumulate(run_max)
    colors = {"run": "#94A3B8", "cumulative": "#164E63"}

    fig, ax = plt.subplots(figsize=(8.3, 5.0), dpi=150)
    ax.plot(
        x,
        run_max,
        color=colors["run"],
        linewidth=1.5,
        linestyle="--",
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="Maximum within each run",
    )
    ax.plot(
        x,
        cumulative,
        color=colors["cumulative"],
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="Cumulative best-so-far across maturity settings",
    )
    for run, raw_value in zip(ordered, run_max):
        suffix = "*" if not run["complete"] else ""
        ax.annotate(
            f"{raw_value:.3f}{suffix}",
            (run["total_maturity"], raw_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#374151",
        )
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xlabel("Configured total_maturity")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{env}: historical-best fitness by total_maturity", loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.text(
        0,
        -0.19,
        "* Current log is incomplete; the plotted value is the best observed so far.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [collect_run(env, total_maturity) for env in ENVS for total_maturity in TOTAL_MATURITIES]
    write_curve_csv(runs, OUT / "best_so_far_by_generation.csv")
    write_summary_csv(runs, OUT / "historical_best_summary.csv")
    for env in ENVS:
        env_runs = [run for run in runs if run["env"] == env]
        plot_learning_curves(env, env_runs, OUT / f"{env}_best_so_far_curves.png")
        plot_maturity_summary(env, env_runs, OUT / f"{env}_historical_best_by_maturity.png")

    for env in ENVS:
        print(env)
        for run in sorted((run for run in runs if run["env"] == env), key=lambda item: item["total_maturity"]):
            status = "complete" if run["complete"] else "incomplete"
            print(
                f"  maturity={run['total_maturity']:>2}: "
                f"best={run['run_max_fit']:.6f}, "
                f"last_budget={run['last_budget']}/{run['max_iters']}, {status}"
            )


if __name__ == "__main__":
    main()
