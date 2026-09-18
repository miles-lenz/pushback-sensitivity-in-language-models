import json
from datetime import datetime, timezone

from tqdm import tqdm

from data import load_gsm8k_dataset
from model import get_model_response, load_model
from prompts import PUSHBACK_PROMPTS, SYSTEM_PROMPT
from schemas import EvaluationRun, ExampleResult, ModelBundle, PushbackResult
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
            run_id=run_id,
            example_id=example_id,
            prompt_name=pushback_name,
            prompt=pushback_prompt,
            messages=messages,
            reference_solution=ref_solution,
            model_bundle=model_bundle,
        )
        example_result.pushbacks[pushback_name] = result

    return example_result


def main() -> None:
    """Main entry point for the pipeline."""

    dataset = load_gsm8k_dataset().select([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print("[INFO] Dataset loaded successfully. Number of examples:", len(dataset))

    model_bundle = load_model("llama")
    print("[INFO] Model and tokenizer loaded successfully.")

    # Create an EvaluationRun object to store all results of the evaluation.
    run_id = f"gsm8k-llama-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    eval_run = EvaluationRun(
        run_id=run_id,
        dataset_name="gsm8k",
        model_name="llama-3.2-3B-Instruct",
    )

    for i, example in enumerate(tqdm(dataset, desc="Evaluating examples")):
        result = evaluate_example(
            run_id=run_id,
            example_id=f"gsm8k-{i:04d}",
            example=example,
            model_bundle=model_bundle,
        )
        eval_run.results.append(result)

    path = f"outputs/{run_id}.json"
    with open(path, "w") as f:
        json.dump(eval_run.model_dump(), f, indent=2)
    print(f"[INFO] Results saved to {path}.")


if __name__ == "__main__":
    main()
