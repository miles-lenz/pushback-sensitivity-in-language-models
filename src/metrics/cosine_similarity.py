import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def compute_run_cosine_similarity(run_json_path: str | Path) -> None:
    """Compute temporal cosine similarity dynamically from full sequence tensors."""
    run_path = Path(run_json_path)

    with open(run_path, "r") as f:
        run_data = json.load(f)

    target_layer = "layer_14"

    for example in run_data.get("results", []):
        initial_act_path = example.get("activations_path")
        if not initial_act_path or not Path(initial_act_path).exists():
            continue

        # Load the full 2D sequence for the initial response
        initial_tensors = torch.load(initial_act_path, weights_only=True)
        initial_seq = initial_tensors[target_layer]

        # Calculate dynamic indices
        i_mid = initial_seq.size(0) // 2
        i_end = initial_seq.size(0) - 1

        for pb_name, pb_data in example.get("pushbacks", {}).values():
            pb_act_path = pb_data.get("activations_path")
            if not pb_act_path or not Path(pb_act_path).exists():
                continue

            pb_tensors = torch.load(pb_act_path, weights_only=True)
            pb_seq = pb_tensors[target_layer]

            p_mid = pb_seq.size(0) // 2
            p_end = pb_seq.size(0) - 1

            similarities = {}
            # Compare Beginning (Index 0)
            sim_beg = F.cosine_similarity(
                initial_seq[0].unsqueeze(0), pb_seq[0].unsqueeze(0), dim=1
            )
            similarities["beginning"] = round(sim_beg.item(), 4)

            # Compare Middle (Dynamic Index)
            sim_mid = F.cosine_similarity(
                initial_seq[i_mid].unsqueeze(0), pb_seq[p_mid].unsqueeze(0), dim=1
            )
            similarities["middle"] = round(sim_mid.item(), 4)

            # Compare End (Dynamic Index)
            sim_end = F.cosine_similarity(
                initial_seq[i_end].unsqueeze(0), pb_seq[p_end].unsqueeze(0), dim=1
            )
            similarities["end"] = round(sim_end.item(), 4)

            pb_data["cosine_similarity"] = similarities

    with open(run_path, "w") as f:
        json.dump(run_data, f, indent=2)

    print(f"[INFO] Temporal similarities computed and saved to {run_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        compute_run_cosine_similarity(sys.argv[1])
    else:
        print("Usage: uv run src/metrics/cosine_similarity.py outputs/<run_id>.json")
