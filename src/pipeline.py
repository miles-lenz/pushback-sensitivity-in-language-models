import json

from tqdm import tqdm

from data import load_gsm8k_dataset
from model import get_model_response, load_model
from prompts import PUSHBACK_PROMPTS, SYSTEM_PROMPT
from schemas import EvaluationRun, ExampleResult, ModelBundle, PushbackResult
from utils import extract_adversarial_answer, extract_answer, save_activations

# NOTES
# - check how to improve performance and if cluster is needed
# - add error handling / catch errors
#   - extract_answer might be none for the inital reference solution
# - add checkpoints / update json in batches not only at end
# - save_actions not good that layer/token selection is hard coded inside
#   - should also be in result JSON, no?
# - move pushback prompt formatting somehwere else? helper function?


def evaluate_pushback(
    name: str,
    prompt: str,
    messages: list[dict],
    reference_solution: str,
    model_bundle: ModelBundle,
    activations_stem: str,
) -> PushbackResult:
    """Evaluate a single pushback prompt."""

    if name == "adversarial":
        adversarial_answer = extract_adversarial_answer(reference_solution)
        prompt = prompt.format(num=adversarial_answer)

    messages_copy = messages + [{"role": "user", "content": prompt}]
    model_response, activations = get_model_response(model_bundle, messages_copy)

    activations_path = save_activations(activations, activations_stem)

    result = PushbackResult(
        model_solution=model_response,
        model_answer=extract_answer(model_response),
        activations_path=activations_path,
    )

    return result


def evaluate_example(
    example_id: int,
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
    activations_path = save_activations(activations, f"{example_id}_main")

    example_result = ExampleResult(
        question=question,
        reference_solution=ref_solution,
        reference_answer=extract_answer(ref_solution),
        model_solution=model_response,
        model_answer=extract_answer(model_response),
        activations_path=activations_path,
    )

    for pushback_name, pushback_prompt in PUSHBACK_PROMPTS.items():
        result = evaluate_pushback(
            name=pushback_name,
            prompt=pushback_prompt,
            messages=messages,
            reference_solution=ref_solution,
            model_bundle=model_bundle,
            activations_stem=f"{example_id}_{pushback_name}",
        )
        example_result.pushbacks[pushback_name] = result

    return example_result


def main() -> None:
    """Main entry point for the pipeline."""

    dataset = load_gsm8k_dataset().select([0, 1])
    print("[INFO] Dataset loaded successfully. Number of examples:", len(dataset))

    model_bundle = load_model("llama")
    print("[INFO] Model and tokenizer loaded successfully.")

    # Create an EvaluationRun object to store all results of the evaluation.
    eval_run = EvaluationRun(dataset_name="gsm8k", model_name="llama-3.2-3B-Instruct")

    for i, example in enumerate(tqdm(dataset, desc="Evaluating examples")):
        id_ = f"gsm8k-llama-{i}"
        result = evaluate_example(id_, example, model_bundle)
        eval_run.results.append(result)
    print("[INFO] Evaluation completed. Number of results:", len(eval_run.results))

    path = "outputs/test.json"
    with open(path, "w") as f:
        json.dump(eval_run.model_dump(), f, indent=2)
    print(f"[INFO] Results saved to {path}.")


if __name__ == "__main__":
    main()
