import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tabulate import SEPARATING_LINE, tabulate
from tqdm import tqdm

from metrics import compute_metrics, parse_args
from plot_config import PUSHBACK_COLORS, TABULATE_CONFIG, apply_plot_config


def run_bootstrap(run_id: str, results: list[dict], iterations: int = 1000) -> dict:
    """
    Performs bootstrap resampling on evaluation results to calculate and print the mean, 
    standard error, and 95% confidence intervals for target metrics.
    """

    target_metrics = [
        "correction_rate",
        "destabilization_rate",
        "confident_wrong_revision_rate",
    ]
    data = {m: defaultdict(list) for m in target_metrics}

    for _ in tqdm(range(iterations)):
        sample = random.choices(results, k=len(results))
        sample_metrics = compute_metrics(sample)

        for pb in ["weak", "medium", "adversarial"]:
            for m in target_metrics:
                data[m][pb].append(sample_metrics[pb][m])

    table = [["metric", "pushback", "mean (%)", "SE (%)", "CI (%)"]]
    for i, (metric_name, metric_data) in enumerate(data.items()):
        # Add separating lines between metrics to increase clarity.
        if i > 0:
            table.append(SEPARATING_LINE)

        for pb_name, values in metric_data.items():
            mean = np.mean(values) * 100
            se = np.std(values) * 100

            ci_lower = np.percentile(values, 2.5) * 100
            ci_upper = np.percentile(values, 97.5) * 100
            ci = f"[{ci_lower:.1f}, {ci_upper:.1f}]"

            table.append([metric_name, pb_name, mean, se, ci])
            metric_name = ""  # Hide metric name for remaining rows.

    print(f"\nBootstrap Results for run '{run_id}' ({iterations=}):")
    print(tabulate(table, **TABULATE_CONFIG))

    return data


def plot_results(data: dict, run_id: str) -> None:
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
            mean = np.mean(vals)
            means.append(mean)
            yerr_lower.append(mean - np.percentile(vals, 2.5))
            yerr_upper.append(np.percentile(vals, 97.5) - mean)
            
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=[yerr_lower, yerr_upper],
            label=pb.title(),
            color=PUSHBACK_COLORS[pb]
        )

    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.legend()
    
    out_path = Path("outputs") / run_id / "plots" / "bootstrap.png"
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

    data = run_bootstrap(run_id, results)
    plot_results(data, run_id)


if __name__ == "__main__":
    args = parse_args()
    main(args.run_id)
