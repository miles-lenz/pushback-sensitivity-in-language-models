"""Aggregate the stats and metrics from multiple runs into tables."""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tabulate import SEPARATING_LINE, tabulate

from plot_config import (
    BAR_STYLE,
    LINE_STYLE,
    PUSHBACK_COLORS,
    STAT_COLORS,
    TABULATE_CONFIG,
    apply_plot_config,
)
from utils import load_json


def get_results() -> tuple[list, list]:
    """Fetch stats and metrics from all runs in the outputs folder."""
    stats, metrics = [], []

    for run in Path("outputs/").iterdir():
        if not run.is_dir():
            continue

        pattern = r"final_\d+"
        if not re.fullmatch(pattern, run.name):
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
            std = (
                np.std(values, ddof=1) * 100
            )  # Added ddof=1 for sample standard deviation

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
            std = (
                np.std(values, ddof=1) * 100
            )  # Added ddof=1 for sample standard deviation

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


def plot_aggregated_stats(stats: list) -> None:
    """Plot the aggregated stats across multiple runs."""
    apply_plot_config()

    pushbacks = ["initial", "weak", "medium", "adversarial"]
    x = np.arange(len(pushbacks))
    fig, ax = plt.subplots()

    # Calculate Accuracy
    acc_means = np.array(
        [np.mean([s[pb]["accuracy"] * 100 for s in stats]) for pb in pushbacks]
    )
    acc_stds = np.array(
        [np.std([s[pb]["accuracy"] * 100 for s in stats], ddof=1) for pb in pushbacks]
    )

    ax.errorbar(
        x,
        acc_means,
        yerr=acc_stds,
        label="Accuracy",
        color=STAT_COLORS.get("accuracy", "#333333"),
        **LINE_STYLE,
    )

    # Calculate Format Success Rate (Equivalent to extracted_rate)
    succ_means = np.array(
        [np.mean([(1 - s[pb]["none_rate"]) * 100 for s in stats]) for pb in pushbacks]
    )
    succ_stds = np.array(
        [
            np.std([(1 - s[pb]["none_rate"]) * 100 for s in stats], ddof=1)
            for pb in pushbacks
        ]
    )

    ax.errorbar(
        x,
        succ_means,
        yerr=succ_stds,
        label="Format Success Rate",
        color=STAT_COLORS.get("extracted_rate", "#333333"),
        **LINE_STYLE,
    )

    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([pb.title() for pb in pushbacks])
    ax.legend(loc="lower left")

    out_path = Path("outputs") / "aggregated_stats.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_aggregated_metrics(metrics: list) -> None:
    """Plot the aggregated metrics across multiple runs."""
    apply_plot_config()

    pushbacks = ["weak", "medium", "adversarial"]
    metric_keys = [
        "correction_rate",
        "destabilization_rate",
        "confident_wrong_revision_rate",
    ]
    metric_labels = [
        "Correction Rate",
        "Destabilization Rate",
        "Confident Wrong\nRevision Rate",
    ]

    fig, ax = plt.subplots()
    x = np.arange(len(metric_keys))
    width = 0.25

    for i, pb in enumerate(pushbacks):
        means = np.array(
            [np.mean([m[pb][k] * 100 for m in metrics]) for k in metric_keys]
        )
        stds = np.array(
            [np.std([m[pb][k] * 100 for m in metrics], ddof=1) for k in metric_keys]
        )

        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            label=pb.title(),
            color=PUSHBACK_COLORS[pb],
            **BAR_STYLE,
        )

    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()

    out_path = Path("outputs") / "aggregated_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Fetch results, print tabular summaries, and save plots."""
    stats, metrics = get_results()

    if not stats or not metrics:
        print("[ERROR] Stats and/or metrics are empty.")
        return

    display_stats_table(stats)
    display_metrics_table(metrics)

    plot_aggregated_stats(stats)
    plot_aggregated_metrics(metrics)


if __name__ == "__main__":
    main()
