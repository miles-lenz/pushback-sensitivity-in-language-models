import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


def run_pca_analysis(run_json_path: str | Path) -> None:
    """Project targeted pushback vectors into 2D space, indicating if the answer changed."""
    run_path = Path(run_json_path)
    with open(run_path, "r") as f:
        run_data = json.load(f)

    vectors = []
    labels = []
    statuses = []

    target_layer = "layer_24"

    for example in run_data.get("results", []):
        # We still need the initial answer to calculate if a change occurred
        initial_answer = example.get("model_answer")
        
        # Load ONLY pushback activations
        for pb_name, pb_data in example.get("pushbacks", {}).items():
            pb_path = pb_data.get("activations_path")
            pb_answer = pb_data.get("model_answer")
            
            if pb_path and Path(pb_path).exists() and initial_answer is not None and pb_answer is not None:
                tensors = torch.load(pb_path, weights_only=True)
                raw_tensor = tensors[target_layer]
                
                # extraction
                if raw_tensor.ndim > 1:
                    raw_tensor = raw_tensor[-1]
                    
                vec = raw_tensor.to(torch.float32).numpy().reshape(-1)
                vectors.append(vec)
                labels.append(pb_name)
                
                if initial_answer != pb_answer:
                    statuses.append("Changed")
                else:
                    statuses.append("Kept")

    if not vectors:
        print("[ERROR] No activation vectors found.")
        return

    X = np.stack(vectors)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    
    color_map = {"weak": "green", "medium": "orange", "adversarial": "red"}
    marker_map = {"Kept": "o", "Changed": "X"}
    
    for idx in range(len(vectors)):
        lbl = labels[idx]
        stat = statuses[idx]
        
        plt.scatter(
            X_pca[idx, 0], 
            X_pca[idx, 1], 
            c=color_map[lbl], 
            marker=marker_map[stat],
            s=100 if stat == "Changed" else 60, 
            alpha=0.7, 
            edgecolors="k"
        )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Weak', markerfacecolor='green', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Medium', markerfacecolor='orange', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Adversarial', markerfacecolor='red', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='X', color='w', label='Outcome: Answer Changed', markerfacecolor='grey', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Outcome: Answer Kept', markerfacecolor='grey', markersize=8, markeredgecolor='k'),
    ]

    plt.title(
        f"PCA of Model Activations: {target_layer}\n"
        f"(Extracted from the activation after pushback)"
    )
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    output_img_path = run_path.parent / f"{run_path.stem}_{target_layer}_pca_pushbacks_only_last_token.png"
    plt.savefig(output_img_path, bbox_inches="tight")
    print(f"[INFO] PCA plot saved to {output_img_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_pca_analysis(sys.argv[1])
    else:
        print("Usage: uv run src/metrics/pca_answer_change.py outputs/<run_id>.json")