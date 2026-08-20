import re
import threading
from pathlib import Path

import torch


def extract_answer(solution: str) -> float | None:
    """Extract numerical answer from the provided solution."""
    # todo: add more complex regex to find answers in different formats
    try:
        return float(solution.split("####")[-1].strip())
    except (ValueError, IndexError):
        return None


def generate_adversarial_answer(solution: str) -> tuple[float, str]:
    """
    Extract the adversarial answer from the given solution.

    Try these approaches in order to generate the adversarial answer:
    - Use second-to-last tagged solution step.
    - Add 1 to the correct answer.

    Return the adversarial answer and the generation method.
    """

    tagged_steps = re.findall(r"<<([^<>]+)>>", solution)
    if len(tagged_steps) >= 2:
        selected_step = tagged_steps[-2]
        adversarial_answer = selected_step.split("=")[-1]
        return float(adversarial_answer.strip()), "intermediate_step"

    print(
        "[WARNING] Expected at least two tagged solution steps in the solution "
        "to generate adversarial answer. Fallback to perturbation."
    )

    return extract_answer(solution) + 1, "perturbation"


def _async_save(tensor: torch.Tensor, path: Path) -> None:
    """Worker function to save the file in the background."""
    torch.save(tensor, path)


def save_activations(
    activations: object,
    run_id: str,
    example_id: str,
    prompt_name: str,
) -> str:
    """Persist activations in a readable run/example/prompt folder structure."""

    # todo: decide which activations to store
    # Move activations to CPU RAM and ensure we don't track gradients.
    tensor_to_save = activations[-1][-1].detach().cpu()

    output_dir = Path("outputs") / run_id / "activations" / example_id
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{prompt_name}.pt"

    # Use a background thread to do the actual network write.
    threading.Thread(target=_async_save, args=(tensor_to_save, path)).start()

    return path.as_posix()
