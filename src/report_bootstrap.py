import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tabulate import SEPARATING_LINE, tabulate
from tqdm import tqdm

from metrics import compute_metrics, compute_stats, parse_args
from plot_config import (
    BAR_STYLE,
    LINE_STYLE,
    PUSHBACK_COLORS,
    STAT_COLORS,
    TABULATE_CONFIG,
    apply_plot_config,
)


def run_bootstrap(run_id: str, results: list[dict], iterations: int = 100) -> tuple:
    """
    Performs bootstrap resampling on evaluation results to calculate and print the mean,
    standard error, and 95% confidence intervals for target metrics/stats.
    """

    original_stats = compute_stats(results)
    original_metrics = compute_metrics(results)

    target_stats = ["accuracy", "extracted_rate"]
    target_metrics = [
        "correction_rate",
        "destabilization_rate",
        "confident_wrong_revision_rate",
    ]
    stats_data = {s: defaultdict(list) for s in target_stats}
    metrics_data = {m: defaultdict(list) for m in target_metrics}

    for _ in tqdm(range(iterations)):
        sample = random.choices(results, k=len(results))

        sample_stats = compute_stats(sample)
        for pb in ["initial", "weak", "medium", "adversarial"]:
            for s in target_stats:
                stats_data[s][pb].append(sample_stats[pb][s])

        sample_metrics = compute_metrics(sample)
        for pb in ["weak", "medium", "adversarial"]:
            for m in target_metrics:
                metrics_data[m][pb].append(sample_metrics[pb][m])

    table = [["metric", "pushback", "mean (%)", "SE (%)", "CI (%)"]]
    for i, (metric_name, metric_data) in enumerate((stats_data | metrics_data).items()):
        # Add separating lines between metrics to increase clarity.
        if i > 0:
            table.append(SEPARATING_LINE)

        metric_name_display = metric_name
        for pb_name, values in metric_data.items():
            if metric_name in target_stats:
                mean = original_stats[pb_name][metric_name] * 100
            else:
                mean = original_metrics[pb_name][metric_name] * 100

            se = np.std(values, ddof=1) * 100

            ci_lower = np.percentile(values, 2.5) * 100
            ci_upper = np.percentile(values, 97.5) * 100
            ci = f"[{ci_lower:.1f}, {ci_upper:.1f}]"

            table.append([metric_name_display, pb_name, mean, se, ci])
            metric_name_display = ""  # Hide metric name for remaining rows.

    print(f"\nBootstrap Results for run '{run_id}' ({iterations=}):")
    print(tabulate(table, **TABULATE_CONFIG))

    return stats_data, metrics_data


def plot_stats(data: dict, og_stats: dict, run_id: str) -> None:
    """Plot the stats results for the given run."""

    apply_plot_config()

    stats = list(data.keys())
    pushbacks = ["initial", "weak", "medium", "adversarial"]

    x = np.arange(len(pushbacks))
    fig, ax = plt.subplots()

    for s in stats:
        means = []
        ci_lower = []
        ci_upper = []

        for pb in pushbacks:
            vals = np.array(data[s][pb]) * 100
            mean = og_stats[pb][s] * 100

            means.append(mean)
            ci_lower.append(np.percentile(vals, 2.5))
            ci_upper.append(np.percentile(vals, 97.5))

        ax.errorbar(
            x,
            means,
            yerr=[means - np.array(ci_lower), np.array(ci_upper) - means],
            label=s.replace("_", " ").title(),
            color=STAT_COLORS.get(s, "#333333"),
            **LINE_STYLE,
        )

    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([pb.title() for pb in pushbacks])

    ax.legend(loc="lower left")

    out_path = Path("outputs") / run_id / "plots" / "stats.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_metrics(data: dict, og_metrics: dict, run_id: str) -> None:
    """Plot the metric results for the given run."""

    apply_plot_config()

    metrics = list(data.keys())
    pushbacks = ["weak", "medium", "adversarial"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots()

    for i, pb in enumerate(pushbacks):
        means = []
        yerr_lower = []
        yerr_upper = []

        for m in metrics:
            vals = np.array(data[m][pb]) * 100
            mean = og_metrics[pb][m] * 100

            means.append(mean)
            yerr_lower.append(mean - np.percentile(vals, 2.5))
            yerr_upper.append(np.percentile(vals, 97.5) - mean)

        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=[yerr_lower, yerr_upper],
            label=pb.title(),
            color=PUSHBACK_COLORS[pb],
            **BAR_STYLE,
        )

    label_map = {
        "correction_rate": "Correction Rate",
        "destabilization_rate": "Destabilization Rate",
        "confident_wrong_revision_rate": "Confident Wrong\nRevision Rate",
    }

    # ax.set_xlabel("Metric")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)

    ax.set_xticks(x)
    ax.set_xticklabels([label_map[m] for m in metrics])

    ax.legend()

    out_path = Path("outputs") / run_id / "plots" / "metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main(run_id: str) -> None:
    """
    Loads result data from the specified run directory
    and executes the bootstrap analysis.
    """

    run_dir = Path("outputs") / run_id
    if not run_dir.exists():
        print(f"[ERROR] Run '{run_id}' does not exist.")
        return

    results_path = run_dir / "results.jsonl"
    with open(results_path, encoding="utf-8") as f:
        results = [json.loads(item) for item in f]

    og_stats = compute_stats(results)
    og_metrics = compute_metrics(results)

    stats_data, metrics_data = run_bootstrap(run_id, results)
    plot_stats(stats_data, og_stats, run_id)
    plot_metrics(metrics_data, og_metrics, run_id)


if __name__ == "__main__":
    args = parse_args()
    main(args.run_id)
