import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from data import load_gsm8k_dataset
from model import get_model_response, load_model
from prompts import PUSHBACK_PROMPTS, SYSTEM_PROMPT
from schemas import ExampleResult, ModelBundle, PushbackResult
from utils import (
    extract_answer,
    generate_adversarial_answer,
    save_activations,
)

BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, nargs="?", default=None)
    return parser.parse_args()


def store_metadata(path: Path, **metadata: Any) -> None:
    """Store metadata about the run at the given path."""
    if path.exists():
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def get_completed_ids(results_path: Path) -> set:
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
    return completed_ids


def evaluate_batch(
    run_id: str,
    batch_ids: list[str],
    batch_examples: list[dict],
    model_bundle: ModelBundle,
) -> list[ExampleResult]:
    """Evaluates a batch of examples simultaneously."""

    # Prepare initial messages for the entire batch.
    batch_messages = []
    for example in batch_examples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        batch_messages.append(messages)

    # Generate initial responses for the whole batch.
    responses, activations = get_model_response(model_bundle, batch_messages)

    # Create ExampleResult objects.
    example_results = []
    for idx in range(len(batch_examples)):
        example_id = batch_ids[idx]
        example = batch_examples[idx]
        response = responses[idx]

        # Handle batch slicing for activations
        # todo: check what happens here?
        act_slice = (
            [layer_act[idx : idx + 1] for layer_act in activations[-1]]
            if isinstance(activations[-1], tuple)
            else activations[-1][idx : idx + 1]
        )
        activations_path = save_activations(act_slice, run_id, example_id, "initial")

        result = ExampleResult(
            example_id=example_id,
            question=example["question"],
            reference_solution=example["answer"],
            reference_answer=extract_answer(example["answer"]),
            model_solution=response,
            model_answer=extract_answer(response),
            activations_path=activations_path,
        )
        example_results.append(result)

        # Append response to history for upcoming pushbacks.
        batch_messages[idx].append({"role": "assistant", "content": response})

    # 4. Process Pushbacks in batches
    # todo: move this into function
    for pushback_name, pushback_prompt_template in PUSHBACK_PROMPTS.items():
        pushback_batch_messages = []
        batch_adv_strategies = []

        for idx in range(len(batch_examples)):
            prompt = pushback_prompt_template
            adv_strategy = None

            if pushback_name == "adversarial":
                adv_answer, adv_strategy = generate_adversarial_answer(
                    batch_examples[idx]["answer"]
                )
                prompt = prompt.format(num=adv_answer)

            batch_adv_strategies.append(adv_strategy)

            hist_copy = list(batch_messages[idx])
            hist_copy.append({"role": "user", "content": prompt})
            pushback_batch_messages.append(hist_copy)

        # Generate pushback responses for the whole batch
        pb_responses, pb_activations = get_model_response(
            model_bundle, pushback_batch_messages
        )

        # Save pushback results
        for idx in range(len(batch_examples)):
            pb_act_slice = (
                [layer_act[idx : idx + 1] for layer_act in pb_activations[-1]]
                if isinstance(pb_activations[-1], tuple)
                else pb_activations[-1][idx : idx + 1]
            )
            pb_path = save_activations(
                pb_act_slice, run_id, batch_ids[idx], pushback_name
            )

            pb_result = PushbackResult(
                prompt_name=pushback_name,
                model_solution=pb_responses[idx],
                model_answer=extract_answer(pb_responses[idx]),
                activations_path=pb_path,
                adversarial_strategy=batch_adv_strategies[idx],
            )
            example_results[idx].pushbacks[pushback_name] = pb_result

    return example_results


def main(run_id: str | None) -> None:
    """
    Entry point for the pipeline.

    # todo: add a little explanation about run_id
    """

    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    run_path = Path(f"outputs/{run_id}/")
    run_path.mkdir(parents=True, exist_ok=True)

    metadata_path = run_path / "metadata.json"
    results_path = run_path / "results.jsonl"

    # Remove the .select() slice for the full run!
    dataset, dataset_name = load_gsm8k_dataset().select([0, 1])
    print(f"[INFO] Dataset loaded successfully. Number of examples: {len(dataset)}")

    model_bundle, model_name = load_model("llama")
    print("[INFO] Model and tokenizer loaded successfully.")

    store_metadata(
        path=metadata_path,
        run_id=run_id,
        dataset=dataset_name,
        model=model_name,
    )

    # todo: move ID logic into own function
    completed_ids = get_completed_ids(results_path)
    print(f"[INFO] Found {len(completed_ids)} completed examples in run '{run_id}'.")

    # Filter out completed examples.
    pending_examples = []
    pending_ids = []
    for i, example in enumerate(dataset):
        example_id = f"gsm8k-{i:04d}"
        if example_id not in completed_ids:
            pending_examples.append(example)
            pending_ids.append(example_id)

    appended_count = 0
    # todo: check if it is good to open context manager outside. Can we make this more clean?
    with open(results_path, "a", encoding="utf-8") as f:
        for i in tqdm(
            range(0, len(pending_examples), BATCH_SIZE), desc="Evaluating batches"
        ):
            batch_examples = pending_examples[i : i + BATCH_SIZE]
            batch_ids = pending_ids[i : i + BATCH_SIZE]

            batch_results = evaluate_batch(
                run_id=run_id,
                batch_ids=batch_ids,
                batch_examples=batch_examples,
                model_bundle=model_bundle,
            )

            for result in batch_results:
                f.write(result.model_dump_json() + "\n")
                appended_count += 1

            # todo: add comment why we need this
            if appended_count % 20 == 0:
                f.flush()
                os.fsync(f.fileno())

        f.flush()
        os.fsync(f.fileno())

    print(f"Pipeline completed successfully. {appended_count} examples processed.")


if __name__ == "__main__":
    args = parse_args()
    main(run_id=args.run_id)
