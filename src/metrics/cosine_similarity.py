"""
NOTE: Check papers to find best layer:
The Internal State of an LLM Knows When It's Lying
No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes 
"""


import json
from pathlib import Path
import torch
import torch.nn.functional as F

def compute_run_cosine_similarity(run_json_path: str | Path) -> None:
    """Compute cosine similarity between initial and pushback activations for an evaluation run."""
    eval_path = Path(run_json_path)
    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    for example_data in eval_data.get("results", []):
        initial_activ_path = example_data.get("activations_path")
        if not initial_activ_path or not Path(initial_activ_path).exists():
            continue
        
        initial_vec = torch.load(initial_activ_path, weights_only=True)

        for _, pushback_data in example_data.get("pushbacks", {}).items():
            pb_activ_path = pushback_data.get("activations_path")
            if not pb_activ_path or not Path(pb_activ_path).exists():
                continue

            pushback_vec = torch.load(pb_activ_path, weights_only=True)

            sim = F.cosine_similarity(initial_vec.unsqueeze(0), pushback_vec.unsqueeze(0)).item()
            pushback_data["cosine_similarity"] = round(sim, 4)

    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"[INFO] Cosine similarity metrics computed and saved to {eval_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        compute_run_cosine_similarity(sys.argv[1])
    else:
        print("Usage: uv src/metrics/metrics.py outputs/<run_id>.json")