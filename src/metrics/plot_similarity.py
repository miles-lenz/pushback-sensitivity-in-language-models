import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_temporal_similarity(run_json_path: str | Path) -> None:
    """Generate a grouped box plot of temporal cosine similarities."""

    run_path = Path(run_json_path)
    with open(run_path, "r") as f:
        run_data = json.load(f)

    data_rows = []

    # 1. Parse JSON and flatten the data for Pandas
    for example in run_data.get("results", []):
        for pb_name, pb_data in example.get("pushbacks", {}).items():
            cosine_data = pb_data.get("cosine_similarity")

            # Ensure the data exists and is a dictionary
            if isinstance(cosine_data, dict):
                for time_label, sim_score in cosine_data.items():
                    if sim_score is not None:
                        data_rows.append(
                            {
                                "Pushback Intensity": pb_name,
                                "Time Step": time_label.capitalize(),
                                "Cosine Similarity": sim_score,
                            }
                        )

    if not data_rows:
        print("[ERROR] No temporal cosine similarity data found in the JSON.")
        return

    # 2. Convert to DataFrame
    df = pd.DataFrame(data_rows)

    # 3. Configure Plot
    # Enforce a logical order for the x-axis and the legend
    time_order = ["Beginning", "Middle", "End"]
    intensity_order = ["weak", "medium", "adversarial"]
    color_palette = {"weak": "#2ca02c", "medium": "#ff7f0e", "adversarial": "#d62728"}

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 4. Generate Grouped Box Plot
    sns.boxplot(
        data=df,
        x="Time Step",
        y="Cosine Similarity",
        hue="Pushback Intensity",
        order=time_order,
        hue_order=intensity_order,
        palette=color_palette,
        linewidth=1.5,
        fliersize=4,
    )

    # 5. Formatting
    plt.title(
        f"Temporal Cosine Similarity by Pushback Intensity\n({run_path.stem})",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Generation Phase", fontsize=12)
    plt.ylabel("Cosine Similarity Score", fontsize=12)

    # Adjust legend position
    plt.legend(title="Pushback", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # 6. Save Plot
    output_img_path = run_path.parent / f"{run_path.stem}_boxplot.png"
    plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Box plot saved successfully to {output_img_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_temporal_similarity(sys.argv[1])
    else:
        print("Usage: uv run src/metrics/plot_similarity.py outputs/<run_id>.json")
