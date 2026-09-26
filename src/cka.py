import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from prompts import PUSHBACK_PROMPTS
from utils import TARGET_LAYERS


def parse_args() -> argparse.Namespace:
    """Parse command line arguments such as run ID."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    return parser.parse_args()


def load_results(run_id: str) -> list[dict]:
    """Load all valid results containing answers for initial and all pushbacks."""

    results_path = Path("outputs") / run_id / "results.jsonl"
    with open(results_path, encoding="utf-8") as f:
        results = [json.loads(item) for item in f]

    valid_results = []
    for res in results:
        answers = [res.get("model_answer")]
        answers += [pb.get("model_answer") for pb in res.get("pushbacks", {}).values()]

        if all(a is not None for a in answers):
            valid_results.append(res)

    print(f"[INFO] Retained {len(valid_results)} fully-aligned questions.")
    return valid_results


def get_activations_path(result: dict, pb_name: str) -> Path:
    """Get the path to the activations for the current result and pushback name."""

    if pb_name == "initial_prompt":
        return Path(result["activations_path"])

    for pb_data in result["pushbacks"].values():
        path_prompt = Path(pb_data["activations_prompt_path"])
        if path_prompt.stem == pb_name:
            return path_prompt

    raise ValueError(f"Unknown pushback name: {pb_name}")


def load_all_layers(results: list[dict], pb_name: str) -> dict[int, torch.Tensor]:
    """
    Load activations from all results for the given pushback.
    Group activations by layer and stack activations per layer so they
    have the shape (num_samples, 3072).
    """

    layer_tensors = defaultdict(list)
    for result in tqdm(results, desc=pb_name):
        path = get_activations_path(result, pb_name)
        activations = torch.load(path, map_location="cpu", weights_only=True)

        for l in TARGET_LAYERS:
            layer_tensors[l].append(activations[f"layer_{l}"].float())

    return {l: torch.stack(tensors) for l, tensors in layer_tensors.items()}


def compute_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes the Linear CKA between two activation matrices."""

    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)

    dot_product = torch.norm(x_centered.t() @ y_centered, p="fro") ** 2

    norm_X = torch.norm(x_centered.t() @ x_centered, p="fro")
    norm_Y = torch.norm(y_centered.t() @ y_centered, p="fro")

    return (dot_product / (norm_X * norm_Y)).item()


def main(run_id: str) -> None:
    """Entry point to compute CKA for the given run."""

    results = load_results(run_id)

    initial_data = load_all_layers(results, "initial_prompt")
    cka_matrix = np.zeros((len(PUSHBACK_PROMPTS), len(TARGET_LAYERS)))

    for i, pb_name in enumerate(PUSHBACK_PROMPTS.keys()):
        pb_data = load_all_layers(results, f"{pb_name}_prompt")

        for j, layer in enumerate(TARGET_LAYERS):
            score = compute_cka(initial_data[layer], pb_data[layer])
            cka_matrix[i, j] = score

    print(cka_matrix)


if __name__ == "__main__":
    args = parse_args()
    main(args.run_id)
