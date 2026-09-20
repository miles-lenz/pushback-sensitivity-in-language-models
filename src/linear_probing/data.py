import json
import math
from pathlib import Path

import torch as t


def load_results(run_id: str) -> list[dict]:
    """
    Load all valid results for the given run.

    A valid results contains answers for the initial
    model response and all pushbacks.
    """

    results_path = Path("outputs") / run_id / "results.jsonl"
    with open(results_path, encoding="utf-8") as f:
        results = [json.loads(item) for item in f]

    valid_results = []
    for res in results:
        answers = [res["model_answer"]]
        answers += [pb["model_answer"] for pb in res["pushbacks"].values()]

        if all(a is not None for a in answers):
            valid_results.append(res)

    return valid_results


def get_activation_paths(result: dict) -> dict[Path]:
    """..."""

    paths = {"initial": Path(result["activations_path"])}
    for pb_data in result["pushbacks"].values():
        path_prompt = Path(pb_data["activations_prompt_path"])
        paths[path_prompt.stem] = path_prompt

    return paths


def get_activations(
    results: list[dict], pushback_type: str, layer: int
) -> t.Tensor:
    """..."""

    act_type_map = {
        "adversarial": "adversarial_prompt",
        "medium": "medium_prompt",
        "weak": "weak_prompt",
    }
    prompt_file_key = act_type_map.get(pushback_type, pushback_type)

    x, y = [], []
    for res in results:
        initial_ans = res.get("model_answer")
        pb_data = res.get("pushbacks", {}).get(pushback_type)

        if initial_ans is None or pb_data is None:
            continue
        pb_ans = pb_data.get("model_answer")
        if pb_ans is None:
            continue

        # 0 = beibehalten, 1 = geändert
        label = 0 if math.isclose(initial_ans, pb_ans, abs_tol=1e-4) else 1

        prompt_path = Path(pb_data["activations_prompt_path"])
        if not prompt_path.exists():
            continue

        activations = t.load(prompt_path, map_location="cpu", weights_only=True)
        layer_act = activations[f"layer_{layer}"].float()

        x.append(layer_act)
        y.append(label)

    return t.stack(x), t.tensor(y, dtype=t.float)


if __name__ == "__main__":
    res = load_results("official_03")
    x, y = get_activations(res, act_type1="adversarial_prompt", layer=14)
