import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def run_pca_analysis(run_json_path: str | Path) -> None:
    """Project targeted unpooled hidden state vectors into 2D space using PCA and plot."""
    run_path = Path(run_json_path)
    with open(run_path, "r") as f:
        run_data = json.load(f)

    vectors = []
    labels = []
    example_ids = []

    target_layer = "layer_22"

    # 1. Collect all vectors and metadata
    for example in run_data.get("results", []):
        ex_id = example["example_id"]
        
        # Load initial activation
        initial_path = example.get("activations_path")
        if initial_path and Path(initial_path).exists():
            tensors = torch.load(initial_path, weights_only=True)
            # Index [0] extracts the first token's hidden state, 
            # ensuring a constant shape (e.g., 3072) for PCA.
            vec = tensors[target_layer][0].to(torch.float32).numpy()
            vectors.append(vec)
            labels.append("initial")
            example_ids.append(ex_id)

        # Load pushback activations
        for pb_name, pb_data in example.get("pushbacks", {}).items():
            pb_path = pb_data.get("activations_path")
            if pb_path and Path(pb_path).exists():
                tensors = torch.load(pb_path, weights_only=True)
                # Index [0] extracts the first token's hidden state
                vec = tensors[target_layer][0].to(torch.float32).numpy()
                vectors.append(vec)
                labels.append(pb_name)
                example_ids.append(ex_id)

    if not vectors:
        print("[ERROR] No activation vectors found.")
        return

    # 2. Convert to 2D numpy array (Samples x Hidden_Dim)
    X = np.stack(vectors)

    # 3. Fit PCA and transform to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 4. Visualization
    plt.figure(figsize=(10, 8))
    unique_labels = ["initial", "weak", "medium", "adversarial"]
    colors = {"initial": "blue", "weak": "green", "medium": "orange", "adversarial": "red"}

    for unique_label in unique_labels:
        idx = [i for i, label in enumerate(labels) if label == unique_label]
        if idx:
            plt.scatter(
                X_pca[idx, 0], 
                X_pca[idx, 1], 
                c=colors[unique_label], 
                label=unique_label, 
                alpha=0.7, 
                edgecolors="k"
            )

    plt.title(f"PCA of Model Activations: {target_layer} ({run_path.stem})")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Save the plot
    output_img_path = run_path.parent / f"{run_path.stem}_layer_{target_layer}_pca.png"
    plt.savefig(output_img_path)
    print(f"[INFO] PCA plot saved to {output_img_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_pca_analysis(sys.argv[1])
    else:
        print("Usage: uv run src/metrics/pca.py outputs/<run_id>.json")