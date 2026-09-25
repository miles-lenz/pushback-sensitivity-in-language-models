import hashlib
import json
import logging
import re
from fractions import Fraction
from pathlib import Path

import torch

logger = logging.getLogger("pipeline")

TARGET_LAYERS = [14, 18, 22, 27]


def extract_answer(solution: str, example_id: str) -> float | None:
    """
    Extract numerical answer from the provided model or reference solution.

    This function is guaranteed to return a float for all reference solutions from
    the GSM8k dataset. A model response might yield None if the answer is not present
    or not properly formatted.
    """
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
        logger.warning(
            f"Failed extraction for ID '{example_id}'. Tail: {solution[-50:]!r}"
        )
        return None


def generate_adversarial_answer(
    ref_solution: str, example_id: str
) -> tuple[float, str]:
    """
    Extract the adversarial answer from the given reference solution.

    We try these approaches in order to generate the adversarial answer:
    - Use second-to-last tagged solution step.
    - Add 1 to the correct answer.

    Return the adversarial answer and the generation method.
    """

    tagged_steps = re.findall(r"<<([^<>]+)>>", ref_solution)
    if len(tagged_steps) >= 2:
        selected_step = tagged_steps[-2]

        adversarial_answer = selected_step.split("=")[-1].strip()
        try:
            adversarial_answer = float(adversarial_answer)
        except ValueError:
            # If it fails, assume it is a fraction.
            adversarial_answer = float(Fraction(adversarial_answer))

        return adversarial_answer, "intermediate_step"

    logger.warning(
        f"ID '{example_id}': Expected at least two tagged solution steps for adversarial answer. "
        "Falling back to perturbation (+1)."
    )

    # Since we only call this function with reference solution from the dataset,
    # we know for sure that we won't get a None value. Therefore, we can safely
    # apply the perturbation.
    return extract_answer(ref_solution, example_id) + 1, "perturbation"


def generate_id(text: str) -> str:
    """Generate a deterministic, 8-character ID based on the given text."""
    hash_object = hashlib.sha256(text.encode("utf-8"))
    return hash_object.hexdigest()[:8]


def extract_activation(
    step_hidden_states: tuple, batch_index: int, token_index: int = -1
) -> dict:
    """
    Extract the activations for specific layers and a specific token from a single generation step.

    Args:
        step_hidden_states: Tuple of hidden states from ONE step of the model.
        batch_index: Which sequence in the batch to extract.
        token_index: Which token to extract (default is -1, the last token).

    Returns:
        A dictionary mapping the layer index to its extracted 1D tensor on the CPU.
    """
    extracted_activations = {}
    for layer in TARGET_LAYERS:
        layer_tensor = (
            step_hidden_states[layer][batch_index, token_index, :].detach().cpu()
        )
        layer_name = f"layer_{layer}"
        extracted_activations[layer_name] = layer_tensor

    return extracted_activations


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
    torch.save(tensor_to_save, path)

    return path.as_posix()


def load_json(path: str | Path) -> dict:
    """Load the JSON file at the given path."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_as_json(data: dict, path: str | Path) -> None:
    """Save the data as a JSON at the given path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
