import argparse
import json
import logging
import math
import os
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv
from tqdm import tqdm

from data import load_gsm8k_dataset
from model import MODEL_CONFIG, get_model_response, load_model
from prompts import PUSHBACK_PROMPTS, SYSTEM_PROMPTS
from schemas import ExampleResult, ModelBundle, PushbackResult
from utils import (
    TARGET_LAYERS,
    extract_activation,
    extract_answer,
    generate_adversarial_answer,
    generate_id,
    save_activations,
)

load_dotenv()

logger = logging.getLogger("pipeline")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
logger.info(f"Using batch_size of {BATCH_SIZE}")

SYSTEM_PROMPT_VERSION = os.getenv("SYSTEM_PROMPT_VERSION", "v1")
SYSTEM_PROMPT = SYSTEM_PROMPTS[SYSTEM_PROMPT_VERSION]
logger.info(f"Using prompt version '{SYSTEM_PROMPT_VERSION}':\n {SYSTEM_PROMPT}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, nargs="?", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def setup_logger(log_file_path: Path) -> logging.StreamHandler:
    """Set up file and console logging, returning the console handler."""
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    return console_handler


def store_metadata(path: Path, **metadata: Any) -> None:
    """Store metadata about the run at the given path."""
    if path.exists():
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def get_completed_ids(results_path: Path) -> set:
    """Create a set with completed IDs from the given results."""
    if not results_path.exists():
        return set()

    completed_ids = set()
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            row = json.loads(line)
            example_id = row.get("example_id")
            if example_id:
                completed_ids.add(example_id)

    logger.info(f"Found {len(completed_ids)} already processed examples.")

    return completed_ids


def get_batches(dataset: Dataset, completed_ids: set) -> Generator:
    """Yields batches of examples that haven't been processed yet."""

    batch = []
    for example in dataset:
        example_id = generate_id(example["question"])
        if example_id in completed_ids:
            continue

        batch.append((example_id, example))
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = []

    # Make sure to also return a partial batch at the end.
    if batch:
        yield batch


def evaluate_batch(
    run_id: str, batch: list, model_bundle: ModelBundle
) -> list[ExampleResult]:
    """Evaluates a batch of examples simultaneously."""

    logger.debug(f"Starting evaluation for batch of size {len(batch)}.")

    # Prepare initial messages for the entire batch.
    batch_messages = []
    for _, example in batch:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        batch_messages.append(messages)

    # Generate initial responses for the whole batch.
    responses, activations = get_model_response(model_bundle, batch_messages)

    # Grab the prefill state (the full prompt) instead of the generated answer.
    initial_prompt_states = activations[0]

    results = []
    for i, (example_id, example) in enumerate(batch):
        response = responses[i]

        tensor = extract_activation(initial_prompt_states, i, token_index=-1)
        activations_path = save_activations(
            tensor, run_id, example_id, "initial_prompt"
        )

        result = ExampleResult(
            example_id=example_id,
            question=example["question"],
            reference_solution=example["answer"],
            reference_answer=extract_answer(example["answer"], example_id),
            model_solution=response,
            model_answer=extract_answer(response, example_id),
            activations_path=activations_path,
        )
        results.append(result)

        # Append response to history for upcoming pushbacks.
        batch_messages[i].append({"role": "assistant", "content": response})

    # Process pushbacks in batches.
    for pb_name, pb_prompt in PUSHBACK_PROMPTS.items():
        logger.debug(f"Applying pushback '{pb_name}' to current batch.")

        pb_batch_messages = []
        batch_adv_strategies = []

        # Build the prompts for the whole batch.
        for i, (_, example) in enumerate(batch):
            adv_strategy = None
            current_pb_prompt = pb_prompt
            if pb_name == "adversarial":
                adv_answer, adv_strategy = generate_adversarial_answer(
                    solution=example["answer"], example_id=example_id
                )
                current_pb_prompt = pb_prompt.format(num=adv_answer)

            batch_adv_strategies.append(adv_strategy)

            # Copy history so pushbacks don't interfere with each other.
            hist_copy = list(batch_messages[i])
            hist_copy.append({"role": "user", "content": current_pb_prompt})
            pb_batch_messages.append(hist_copy)

        # Generate pushback responses for the whole batch.
        pb_responses, pb_activations = get_model_response(
            model_bundle, pb_batch_messages
        )

        # Extract the pre-fill (prompt) step.
        prompt_states = pb_activations[0]

        # Save pushback results and attach them to our ExampleResults.
        for i, (example_id, _) in enumerate(batch):
            # Extract and save the prompt activation (right before generating).
            prompt_tensor = extract_activation(prompt_states, i, token_index=-1)
            prompt_path = save_activations(
                prompt_tensor, run_id, example_id, f"{pb_name}_prompt"
            )

            pb_result = PushbackResult(
                prompt_name=pb_name,
                model_solution=pb_responses[i],
                model_answer=extract_answer(pb_responses[i], example_id),
                activations_prompt_path=prompt_path,
                adversarial_strategy=batch_adv_strategies[i],
            )

            # Attach this pushback result to the parent ExampleResult
            results[i].pushbacks[pb_name] = pb_result

    return results


def main(run_id: str | None, debug: bool = False) -> None:
    """
    Entry point for the pipeline.

    If run_id is provided, the pipeline resumes by checking that folder
    and skipping already completed examples. If None, it starts a fresh run.

    If debug is True, only 2 examples will be processed for quick testing.
    """

    # Use a standard datetime ID if no run ID is provided.
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Create folder to store all outputs for this run.
    run_path = Path(f"outputs/{run_id}/")
    run_path.mkdir(parents=True, exist_ok=True)

    # Define paths within the outputs folder.
    metadata_path = run_path / "metadata.json"
    results_path = run_path / "results.jsonl"
    log_file_path = run_path / "run.log"

    console_handler = setup_logger(log_file_path)
    logger.info(f"Starting a pipeline run with ID: {run_id}")

    dataset, dataset_name = load_gsm8k_dataset()
    if debug:
        logger.info("--debug flag detected. Using fewer examples.")
        dataset = dataset.select(range(2))
    logger.info(f"Dataset loaded successfully. Number of examples: {len(dataset)}")

    model_bundle, model_name = load_model()
    logger.info("Model and tokenizer loaded successfully.")

    store_metadata(
        path=metadata_path,
        run_id=run_id,
        dataset=dataset_name,
        model=model_name,
        target_layers=TARGET_LAYERS,
        prompt_version=SYSTEM_PROMPT_VERSION,
        prompt_template=SYSTEM_PROMPT,
        **MODEL_CONFIG,
    )

    # Use a generator to yield batches for IDs that
    # have not been processed yet.
    completed_ids = get_completed_ids(results_path)
    batches = get_batches(dataset, completed_ids)

    # Mute the console logger right before the loop to not
    # disrupt the tqdm progress bar.
    console_handler.setLevel(logging.CRITICAL)

    # Calculate absolute total and the number of already completed batches.
    total_batches = math.ceil(len(dataset) / BATCH_SIZE)
    completed_batches = math.ceil(len(completed_ids) / BATCH_SIZE)

    # Open the results file once outside the loop to avoid overhead
    # of opening and closing it for every batch.
    with open(results_path, "a", encoding="utf-8") as f:
        for i, batch in tqdm(
            enumerate(batches),
            desc="Evaluating batches",
            total=total_batches,
            initial=completed_batches,
        ):
            batch_results = evaluate_batch(
                run_id=run_id,
                batch=batch,
                model_bundle=model_bundle,
            )
            f.writelines(result.model_dump_json() + "\n" for result in batch_results)

            # Force RAM buffers to write to the physical disk periodically.
            # This ensures we don't lose the whole batch if the script crashes.
            if i % 5 == 0:
                f.flush()
                os.fsync(f.fileno())

        f.flush()
        os.fsync(f.fileno())

    console_handler.setLevel(logging.INFO)
    logger.info("Pipeline completed successfully!\n")


if __name__ == "__main__":
    args = parse_args()
    main(run_id=args.run_id, debug=args.debug)
