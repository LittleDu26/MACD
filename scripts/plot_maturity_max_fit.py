#!/usr/bin/env python3
"""Plot best-so-far fitness over the maximum-maturity sweep.

For each task, a point at maturity m is the largest fitness observed in any
run whose configured maximum maturity is less than or equal to that point.
This makes the displayed curve a historical maximum (running maximum), rather
than a series of independent per-run maxima.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("result/MACD_age_group/MACD_age")
OUTPUT_DIR = ROOT.parent / "plots_max_fit"
TASKS = ("GapJumper-v0", "Thrower-v0", "Walker-v0")


def read_best_fit(task: str) -> list[tuple[int, float]]:
    values = []
    for table_path in ROOT.glob(f"maturity_*/{task}/0/table.csv"):
        max_maturity = int(re.search(r"maturity_(\d+)", str(table_path)).group(1))
        with table_path.open(newline="") as table_file:
            best_fit = max(float(row["fit"]) for row in csv.DictReader(table_file))
        values.append((max_maturity, best_fit))
    return sorted(values)


def plot_task(task: str) -> None:
    values = read_best_fit(task)
    maturities, run_best_fits = zip(*values)
    historical_best_fits = []
    best_so_far = float("-inf")
    for fit in run_best_fits:
        best_so_far = max(best_so_far, fit)
        historical_best_fits.append(best_so_far)
    positions = range(len(maturities))

    fig, axis = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    axis.plot(
        positions,
        historical_best_fits,
        color="#1f5a94",
        marker="o",
        markersize=6,
        linewidth=2.2,
        markerfacecolor="white",
        markeredgewidth=1.8,
    )
    for position, best_fit in zip(positions, historical_best_fits):
        axis.annotate(
            f"{best_fit:.3f}",
            (position, best_fit),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    axis.set_title(
        f"{task}: Historical Best Fitness by Maximum Maturity",
        pad=12,
        weight="bold",
    )
    axis.set_xlabel("Maximum maturity")
    axis.set_ylabel("Historical best fitness")
    axis.set_xticks(list(positions), [str(maturity) for maturity in maturities])
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.margins(x=0.06, y=0.18)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{task.removesuffix('-v0')}_max_fit.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        plot_task(task)


if __name__ == "__main__":
    main()
