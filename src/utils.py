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
    activations: object,
    run_id: str,
    example_id: str,
    prompt_name: str,
) -> str:
    """Persist activations in a readable run/example/prompt folder structure."""

    # todo: decide which activations to store
    activations = activations[-1][-1]

    output_dir = Path("outputs/activations") / run_id / example_id
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{prompt_name}.pt"

    torch.save(activations, path)

    return path.as_posix()
