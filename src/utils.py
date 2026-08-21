import hashlib
import logging
import re
import threading
from pathlib import Path

import torch

logger = logging.getLogger("pipeline")


def extract_answer(solution: str) -> float:
    """Extract numerical answer from the provided solution."""
    try:
        if "####" not in solution:
            raise ValueError

        answer_text = solution.split("####")[-1]

        match = re.search(r"[-+]?[0-9,]+\.?[0-9]*", answer_text)
        if not match:
            raise ValueError

        clean_number_str = match.group(0).replace(",", "")
        return float(clean_number_str)

    except ValueError:
        logger.critical(
            f"Could not extract answer for the following solution:\n{solution}"
        )
        raise


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

    logger.warning(
        "Expected at least two tagged solution steps in the solution "
        "to generate adversarial answer. Fallback to perturbation."
    )

    return extract_answer(solution) + 1, "perturbation"


def generate_id(text: str) -> str:
    """Generate a deterministic, 8-character ID based on the given text."""
    hash_object = hashlib.sha256(text.encode("utf-8"))
    return hash_object.hexdigest()[:8]


def _async_save(tensor: torch.Tensor, path: Path) -> None:
    """Worker function to save the file in the background."""
    torch.save(tensor, path)


def extract_activation(activations: tuple, batch_index: int) -> None:
    """Extract the activations for a single example from a batched model output."""
    # todo: decide which activations to store
    return activations[-1][-1][batch_index : batch_index + 1].detach().cpu()


def save_activations(
    tensor_to_save: object,
    run_id: str,
    example_id: str,
    prompt_name: str,
) -> str:
    """Persist activations in a readable run/example/prompt folder structure."""

    output_dir = Path("outputs") / run_id / "activations" / example_id
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{prompt_name}.pt"

    # Use a background thread to do the actual network write.
    threading.Thread(target=_async_save, args=(tensor_to_save, path)).start()

    return path.as_posix()
