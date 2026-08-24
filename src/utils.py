import re
from pathlib import Path

import torch


def extract_answer(solution: str) -> float | None:
    """Extract numerical answer from the provided solution."""
    # todo: add more complex regex to find answers in different formats
    try:
        return float(solution.split("####")[-1].strip())
    except (ValueError, IndexError):
        return None


def extract_adversarial_answer(solution: str) -> float:
    """Extract the adversarial answer from the second-to-last tagged solution step."""

    tagged_steps = re.findall(r"<<([^<>]+)>>", solution)
    if len(tagged_steps) < 2:
        raise ValueError("Expected at least two tagged solution steps in the solution.")

    selected_step = tagged_steps[-2]
    adversarial_answer = selected_step.split("=")[-1]

    return float(adversarial_answer.strip())


def save_activations(
    activations: tuple,
    run_id: str,
    example_id: str,
    prompt_name: str,
) -> str:
    """Save the full uncompressed sequence of hidden states for targeted layers."""
    
    # Target layers based on literature. Max layers 28
    target_layers = [14, 22, 24]
    layer_tensors = {}
    
    for layer_idx in target_layers:
        # Extract the specific layer for every generation step
        token_vectors = [step[layer_idx][:, -1, :].squeeze(1) for step in activations]
        
        # Stack into a 2D tensor: shape (num_tokens, hidden_dim)
        sequence_tensor = torch.cat(token_vectors, dim=0)
        
        # Save in full float32 precision
        layer_tensors[f"layer_{layer_idx}"] = sequence_tensor.cpu()

    output_dir = Path("outputs/activations") / run_id / example_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prompt_name}.pt"

    torch.save(layer_tensors, path)
    
    return path.as_posix()