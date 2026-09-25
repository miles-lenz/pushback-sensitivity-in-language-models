"""Aggregate the stats and metrics from multiple runs into tables."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from tabulate import SEPARATING_LINE, tabulate

from utils import load_json

TABULATE_CONFIG = {
    "headers": "firstrow",
    "tablefmt": "fancy_outline",
    "floatfmt": ".1f",
}


def get_results() -> tuple[list, list]:
    """Fetch stats and metrics from all runs in the outputs folder."""
    stats, metrics = [], []

    for run in Path("outputs/").iterdir():
        if not run.is_dir():
            continue

        stats_file = run / "stats.json"
        metrics_file = run / "metrics.json"

        if stats_file.exists() and metrics_file.exists():
            stats.append(load_json(stats_file))
            metrics.append(load_json(metrics_file))

    return stats, metrics


def display_stats_table(stats: list) -> None:
    """Display a table to show the aggregated stats."""

    data = defaultdict(dict)
    for pb in ["initial", "weak", "medium", "adversarial"]:
        data["accuracy"][pb] = [s[pb]["accuracy"] for s in stats]
        data["format_failure_rate"][pb] = [s[pb]["none_rate"] for s in stats]

    table = [["metric", "pushback", "mean (%)", "std (%)"]]
    for i, (metric_name, metric_data) in enumerate(data.items()):
        # Add separating lines between metrics to increase clarity.
        if i > 0:
            table.append(SEPARATING_LINE)

        for pb_name, values in metric_data.items():
            mean = np.mean(values) * 100
            std = np.std(values) * 100

            table.append([metric_name, pb_name, mean, std])
            metric_name = ""  # Hide metric name for remaining rows.

    avg_n = int(np.mean([s["total"] for s in stats]))
    print(f"\nGeneral Stats (aggregated from {len(stats)} runs with avg_n={avg_n}):")
    print(tabulate(table, **TABULATE_CONFIG))


def display_metrics_table(metrics: list) -> None:
    """Display a table to show the aggregated metrics."""

    metric_names = [
        "correction_rate",
        "destabilization_rate",
        "confident_wrong_revision_rate",
    ]

    data = defaultdict(dict)
    for pushback in ["weak", "medium", "adversarial"]:
        data["answer_change_rate"][pushback] = [
            m[pushback]["total_changed"] / max(m[pushback]["total"], 1) for m in metrics
        ]
        for metric in metric_names:
            data[metric][pushback] = [m[pushback][metric] for m in metrics]

    table = [["metric", "pushback", "mean (%)", "std (%)"]]
    for i, (metric_name, metric_data) in enumerate(data.items()):
        # Add separating lines between metrics to increase clarity.
        if i > 0:
            table.append(SEPARATING_LINE)

        for pb_name, values in metric_data.items():
            mean = np.mean(values) * 100
            std = np.std(values) * 100

            table.append([metric_name, pb_name, mean, std])
            metric_name = ""  # Hide metric name for remaining rows.

    # Compute the average sample sizes per pushback intensity.
    n_weak = int(np.mean([m["weak"]["total"] for m in metrics]))
    n_medium = int(np.mean([m["medium"]["total"] for m in metrics]))
    n_adv = int(np.mean([m["adversarial"]["total"] for m in metrics]))

    print(f"\nChange Metrics (aggregated from {len(metrics)} runs):")
    print(
        "- Cases where the model failed to output a valid formatted answer (None) are excluded.",
        f"\n- Sample sizes (avg): Weak n={n_weak} | Medium n={n_medium} | Adversarial n={n_adv}",
    )
    print(tabulate(table, **TABULATE_CONFIG))


def main() -> None:
    """..."""

    stats, metrics = get_results()

    if not stats or not metrics:
        print("[ERROR] Stats and/or metrics are empty.")
        return

    display_stats_table(stats)
    display_metrics_table(metrics)


if __name__ == "__main__":
    main()
