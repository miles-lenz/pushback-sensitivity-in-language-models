import argparse
import json
import os
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from data import load_gsm8k_dataset
from model import get_model_response, load_model
from prompts import PUSHBACK_PROMPTS, SYSTEM_PROMPT
from schemas import ExampleResult, ModelBundle, PushbackResult
from utils import extract_adversarial_answer, extract_answer, save_activations


def evaluate_pushback(
    run_id: str,
    example_id: str,
    prompt_name: str,
    prompt: str,
    messages: list[dict],
    reference_solution: str,
    model_bundle: ModelBundle,
) -> PushbackResult:
    """Evaluate a single pushback prompt."""

    if prompt_name == "adversarial":
        adversarial_answer = extract_adversarial_answer(reference_solution)
        prompt = prompt.format(num=adversarial_answer)

    messages_copy = messages + [{"role": "user", "content": prompt}]
    model_response, activations = get_model_response(model_bundle, messages_copy)

    activations_path = save_activations(activations, run_id, example_id, prompt_name)

    result = PushbackResult(
        prompt_name=prompt_name,
        model_solution=model_response,
        model_answer=extract_answer(model_response),
        activations_path=activations_path,
    )

    return result


def evaluate_example(
    run_id: str,
    example_id: str,
    example: dict,
    model_bundle: ModelBundle,
) -> ExampleResult:
    """Evaluate a single example from the dataset."""

    question, ref_solution = example["question"], example["answer"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    model_response, activations = get_model_response(model_bundle, messages)

    messages.append({"role": "assistant", "content": model_response})
    activations_path = save_activations(activations, run_id, example_id, "initial")

    example_result = ExampleResult(
        example_id=example_id,
        question=question,
        reference_solution=ref_solution,
        reference_answer=extract_answer(ref_solution),
        model_solution=model_response,
        model_answer=extract_answer(model_response),
        activations_path=activations_path,
    )

    for pushback_name, pushback_prompt in PUSHBACK_PROMPTS.items():
        result = evaluate_pushback(
            prompt_name=pushback_name,
            prompt=pushback_prompt,
            messages=messages,
            reference_solution=ref_solution,
            model_bundle=model_bundle,
            run_id=run_id,
            example_id=example_id,
        )
        example_result.pushbacks[pushback_name] = result

    return example_result


def parse_args() -> Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        nargs="?",
        default=None,
    )

    return parser.parse_args()


def get_completed_ids(results_path: Path) -> set:
    """..."""

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


def main(run_id: str | None) -> None:
    """Run a new pipeline or resume a pipeline if run_id is set."""

    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    run_path = Path(f"outputs/{run_id}/")
    run_path.mkdir(parents=True, exist_ok=True)

    metadata_path = run_path / "metadata.json"
    results_path = run_path / "results.jsonl"

    dataset = load_gsm8k_dataset().select([0, 1, 2, 3])
    print("[INFO] Dataset loaded successfully. Number of examples:", len(dataset))

    model_bundle = load_model("llama")
    print("[INFO] Model and tokenizer loaded successfully.")

    # Store metadata about the run if they not already exist.
    if not metadata_path.exists():
        metadata = {
            "run_id": run_id,
            "dataset_name": "gsm8k",
            "model_name": "llama-3.2-3B-Instruct",
        }
        with open(run_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    completed_ids = get_completed_ids(results_path)
    print(f"[INFO] Found {len(completed_ids)} completed examples in run '{run_id}'.")

    appended_count = 0
    with open(run_path / "results.jsonl", "a", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(dataset, desc="Evaluating examples")):
            example_id = f"gsm8k-{i:04d}"

            # Skip existing examples to save resources.
            if example_id in completed_ids:
                continue

            result = evaluate_example(
                run_id=run_id,
                example_id=example_id,
                example=example,
                model_bundle=model_bundle,
            )
            f.write(result.model_dump_json() + "\n")
            appended_count += 1

            # Force buffered JSONL writes to disk periodically so a crash loses at most this batch.
            if appended_count % 20 == 0:
                f.flush()
                os.fsync(f.fileno())

        # Ensure the final partial batch is also durably written.
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    args = parse_args()
    main(run_id=args.run_id)
