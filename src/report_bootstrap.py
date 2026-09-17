import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from tabulate import SEPARATING_LINE, tabulate
from tqdm import tqdm

from metrics import compute_metrics, parse_args
from report_aggregate import TABULATE_CONFIG


def run_bootstrap(run_id: str, results: list[dict], iterations: int = 100) -> None:
    """..."""

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


def main(run_id: str) -> None:
    """..."""

    run_dir = Path("outputs") / run_id
    if not run_dir.exists():
        print(f"[ERROR] Run '{run_id}' does not exist.")
        return

    results_path = run_dir / "results.jsonl"
    with open(results_path, encoding="utf-8") as f:
        results = [json.loads(item) for item in f]

    run_bootstrap(run_id, results)


if __name__ == "__main__":
    args = parse_args()
    main(args.run_id)
