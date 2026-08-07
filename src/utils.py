import hashlib
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


def save_activations(activations: object, stem: str) -> str:
    """Persist activations to a binary file and return the file path."""

    activations = activations[-1][-1]

    output_dir = Path("outputs/activations")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_hash = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
    path = output_dir / f"{file_hash}.pt"

    torch.save(activations, path)

    return str(path)
