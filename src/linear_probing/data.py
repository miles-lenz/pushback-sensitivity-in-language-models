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


def get_activations(results: list[dict], pushback_type: str, layer: int) -> t.Tensor:
    """..."""
    x, y = [], []
    for res in results:
        initial_ans = res.get("model_answer")
        pb_data = res.get("pushbacks", {}).get(pushback_type)
        pb_ans = pb_data.get("model_answer")

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


def get_specific_activations(
    results: list[dict],
    pushback_type: str,
    layer: int,
    num: int = 50,
    changes_only: bool = True,
) -> tuple[t.Tensor, list[dict]]:
    """Extrahiert gezielt `num` Aktivierungen (z. B. nur eingeknickte Antworten y=1)

    sowie die dazugehörigen Beispieldaten für spätere Interventionen.
    """
    x = []
    selected_results = []

    for res in results:
        initial_ans = res.get("model_answer")
        pb_data = res.get("pushbacks", {}).get(pushback_type)

        if initial_ans is None or pb_data is None:
            continue

        pb_ans = pb_data.get("model_answer")
        if pb_ans is None:
            continue

        # Prüfen, ob die Antwort gekippt ist (y = 1)
        has_changed = not math.isclose(initial_ans, pb_ans, abs_tol=1e-4)

        # Filter: Überspringe unveränderte Antworten (y = 0)
        if changes_only and not has_changed:
            continue

        prompt_path = Path(pb_data["activations_prompt_path"])
        if not prompt_path.exists():
            continue

        activations = t.load(prompt_path, map_location="cpu", weights_only=True)
        layer_act = activations[f"layer_{layer}"].float()

        x.append(layer_act)
        selected_results.append(res)

        if len(x) == num:
            break

    if len(x) < num:
        print(f"[WARNUNG] Nur {len(x)} passende Beispiele gefunden (gefordert: {num}).")

    x_tensor = t.stack(x)
    return x_tensor, selected_results


if __name__ == "__main__":
    res = load_results("official_03")
    x, y = get_activations(res, act_type1="adversarial_prompt", layer=14)
