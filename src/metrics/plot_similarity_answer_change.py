import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_cosine_similarity(run_json_path: str | Path) -> None:
    """Visualize cosine similarity distributions grouped by answer stability."""
    
    run_path = Path(run_json_path)
    with open(run_path, "r") as f:
        run_data = json.load(f)

    records = []

    # 1. Parse JSON and classify outcomes
    for example in run_data.get("results", []):
        initial_answer = example.get("model_answer")
        
        for pb_name, pb_data in example.get("pushbacks", {}).items():
            pb_answer = pb_data.get("model_answer")
            sim_dict = pb_data.get("cosine_similarity")
            
            if not sim_dict or initial_answer is None or pb_answer is None:
                continue
                
            # Determine if the model changed its mathematical answer
            answer_changed = "Changed" if initial_answer != pb_answer else "Kept"
            
            # Extract the three temporal points
            for time_label in ["beginning", "middle", "end"]:
                if time_label in sim_dict:
                    records.append({
                        "Pushback_Type": pb_name,
                        "Time_Point": time_label.capitalize(),
                        "Cosine_Similarity": sim_dict[time_label],
                        "Answer_Status": answer_changed
                    })

    if not records:
        print("[ERROR] No valid cosine similarity data found to plot.")
        return

    df = pd.DataFrame(records)

    # 2. Generate the Box Plot
    plt.figure(figsize=(10, 6))
    
    # sns.boxplot automatically groups by x (Time_Point) and separates by hue (Answer_Status)
    sns.boxplot(
        data=df, 
        x="Time_Point", 
        y="Cosine_Similarity", 
        hue="Answer_Status",
        palette={"Kept": "#2ca02c", "Changed": "#d62728"}, # Green for kept, Red for changed
        order=["Beginning", "Middle", "End"]
    )

    plt.title(f"Cosine Similarity by Answer Stability ({run_path.stem})", pad=15)
    plt.xlabel("Generation Stage")
    plt.ylabel("Cosine Similarity (Initial vs. Pushback)")
    plt.ylim(0, 1)
    plt.legend(title="Model Answer")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Save the plot
    output_img_path = run_path.parent / f"{run_path.stem}_cosine_boxplot.png"
    plt.savefig(output_img_path, bbox_inches="tight")
    print(f"[INFO] Visualization saved to {output_img_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_cosine_similarity(sys.argv[1])
    else:
        print("Usage: uv run src/metrics/visualize_cosine.py outputs/<run_id>.json")