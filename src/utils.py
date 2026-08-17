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
    """Mean-pool last layer hidden states across all generated tokens and save."""
    token_vectors = [step[-1][:, -1, :].squeeze(1) for step in activations]

    sequence_tensor = torch.cat(token_vectors, dim=0)

    output_dir = Path("outputs/activations") / run_id / example_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prompt_name}.pt"

    torch.save(sequence_tensor[0], path)
    return path.as_posix()
