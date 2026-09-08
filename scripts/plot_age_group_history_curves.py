#!/usr/bin/env python3
"""Plot best-so-far fitness curves for MACD_age_group.

Each task gets one figure.  Inside a task figure, every configured maximum
maturity is drawn as a separate curve.  Points follow the row order in
table.csv: row 1 is evaluation 1, row 2 is evaluation 2, and so on.  The y
value is the running historical maximum of fit up to that evaluation.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("result/MACD_age_group/MACD_age")
OUTPUT_DIR = ROOT.parent / "plots_history_max_fit"
SUMMARY_CSV = OUTPUT_DIR / "history_max_fit_points.csv"

MATURITY_RE = re.compile(r"maturity_(\d+)")
EXCLUDED_MATURITIES = {15}
COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#4E79A7",
    "#F28E2B",
)
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "*")


def discover_tables() -> dict[str, list[tuple[int, Path]]]:
    tables: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for table_path in ROOT.glob("maturity_*/*/0/table.csv"):
        match = MATURITY_RE.search(str(table_path))
        if not match:
            continue
        task = table_path.parents[1].name
        max_maturity = int(match.group(1))
        if max_maturity in EXCLUDED_MATURITIES:
            continue
        tables[task].append((max_maturity, table_path))

    return {
        task: sorted(paths, key=lambda item: item[0])
        for task, paths in sorted(tables.items())
    }


def read_history_curve(table_path: Path) -> list[tuple[int, float]]:
    history_curve: list[tuple[int, float]] = []
    best_so_far = float("-inf")
    with table_path.open(newline="", encoding="utf-8") as table_file:
        for evaluation_count, row in enumerate(csv.DictReader(table_file), start=1):
            fit = float(row["fit"])
            best_so_far = max(best_so_far, fit)
            history_curve.append((evaluation_count, best_so_far))
    return history_curve


def write_summary(rows: list[dict[str, str]]) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as summary_file:
        fieldnames = [
            "task",
            "max_maturity",
            "evaluation_count",
            "history_max_fit",
            "source_table",
        ]
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_task(task: str, table_paths: list[tuple[int, Path]], rows: list[dict[str, str]]) -> None:
    fig, axis = plt.subplots(figsize=(9.2, 5.6), dpi=180)

    for index, (max_maturity, table_path) in enumerate(table_paths):
        curve = read_history_curve(table_path)
        if not curve:
            continue

        evaluation_counts = [point[0] for point in curve]
        history = [point[1] for point in curve]
        label = f"max maturity {max_maturity}"

        axis.plot(
            evaluation_counts,
            history,
            color=COLORS[index % len(COLORS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            marker=MARKERS[index % len(MARKERS)],
            markersize=3.6,
            markevery=max(1, len(evaluation_counts) // 18),
            linewidth=2.1,
            label=label,
        )

        for evaluation_count, history_max_fit in curve:
            rows.append(
                {
                    "task": task,
                    "max_maturity": str(max_maturity),
                    "evaluation_count": str(evaluation_count),
                    "history_max_fit": f"{history_max_fit:.12g}",
                    "source_table": str(table_path),
                }
            )

    axis.set_title(f"{task}: Historical Max Fitness Curves", loc="left", pad=12, weight="bold")
    axis.set_xlabel("Evaluation count")
    axis.set_ylabel("Historical max fitness")
    axis.grid(True, color="#D9DEE7", linewidth=0.75, alpha=0.9)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=2, fontsize=8.5)
    axis.margins(x=0.02, y=0.12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{task.removesuffix('-v0')}_history_max_fit_curves.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    tables = discover_tables()
    for task, table_paths in tables.items():
        plot_task(task, table_paths, rows)
    write_summary(rows)
    print(f"Wrote {len(tables)} task plots to {OUTPUT_DIR}")
    print(f"Wrote curve data to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
