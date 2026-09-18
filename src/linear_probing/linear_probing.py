import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch


def find_data_directory(repo_root: Path) -> tuple[Path, Path]:
    results_dir = repo_root / "results"
    if not results_dir.exists():
        results_dir = repo_root

    # Get the first jsonl file
    jsonl_candidates = list(results_dir.rglob("*.jsonl"))
    if not jsonl_candidates:
        raise FileNotFoundError(f"Keine .jsonl Datei in {results_dir} gefunden.")

    jsonl_path = jsonl_candidates[0]
    base_output_dir = jsonl_path.parent
    return jsonl_path, base_output_dir


def resolve_activation_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path or not isinstance(raw_path, str):
        return None

    clean_path = raw_path.replace("\\", "/").strip("/")

    # Check if the path has acitvations
    if "activations" in clean_path:
        rel_sub = clean_path[clean_path.index("activations") :]
        resolved = base_dir / rel_sub
        if resolved.is_file():
            return resolved

    # Fallback of above...
    target_name = Path(clean_path).name
    parent_id = Path(clean_path).parent.name
    fallback = base_dir / "activations" / parent_id / target_name
    if fallback.is_file():
        return fallback

    return None


def load_dataset(
    jsonl_path: Path,
    activations_base_dir: Path,
    pushback_type: str = "medium",
    use_prompt_activations: bool = True,
):
    """Load X and y. X is the activation vectors and y the labels (1=changed, 0=stable)."""
    layers = ["layer_14", "layer_18", "layer_22", "layer_27"]
    layer_data = {l: [] for l in layers}
    targets = []
    missing_files = 0

    path_key = (
        "activations_prompt_path"
        if use_prompt_activations
        else "activations_answer_path"
    )

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            initial_ans = entry.get("model_answer")
            pushbacks = entry.get("pushbacks", {})

            if pushback_type not in pushbacks or initial_ans is None:
                continue

            pb_entry = pushbacks[pushback_type]
            pb_ans = pb_entry.get("model_answer")
            if pb_ans is None:
                continue

            changed = int(not np.isclose(initial_ans, pb_ans, atol=1e-4))

            raw_pt_path = pb_entry.get(path_key)
            pt_path = resolve_activation_path(activations_base_dir, raw_pt_path)

            if pt_path is None:
                missing_files += 1
                continue

            try:
                act_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
                for l in layers:
                    vec = act_dict[l].to(torch.float32).numpy()
                    layer_data[l].append(vec)

                targets.append(changed)
            except Exception as e:
                print(f"Fehler beim Lesen von {pt_path}: {e}")
                missing_files += 1
                continue

    if missing_files > 0:
        print(f"Warnung: {missing_files} Dateien für '{path_key}' nicht gefunden.")

    X_by_layer = {l: np.array(vecs) for l, vecs in layer_data.items()}
    y = np.array(targets)
    return X_by_layer, y


def evaluate_probe_train_test(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2
) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    # 80/20 train, test - Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    # Compute probabilites for AUROC.
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    train_auroc = roc_auc_score(y_train, y_train_proba)
    test_auroc = roc_auc_score(y_test, y_test_proba)

    return float(train_auroc), float(test_auroc)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    jsonl_path, activations_dir = find_data_directory(repo_root)

    print(f"Verwende Daten aus: {jsonl_path.name}")

    for pb_type in ["weak", "medium", "adversarial"]:
        print(f"\n================ Pushback: {pb_type} ================")
        X_by_layer, y = load_dataset(
            jsonl_path,
            activations_dir,
            pushback_type=pb_type,
            use_prompt_activations=True,
        )

        if len(y) == 0:
            print("Keine gültigen Datenpunkte gefunden.")
            continue

        n_changed = int(np.sum(y))
        n_stable = len(y) - n_changed
        n_train = int(len(y) * 0.8)
        n_test = len(y) - n_train

        print(
            f"Total: {len(y)} (Train: {n_train}, Test: {n_test}) | Changed (y=1): {n_changed} | Stable (y=0): {n_stable}"
        )

        for layer_name, X in X_by_layer.items():
            train_auroc, test_auroc = evaluate_probe_train_test(X, y, test_size=0.2)
            print(
                f"  {layer_name}: Train AUROC = {train_auroc:.4f} | Test AUROC = {test_auroc:.4f}"
            )


if __name__ == "__main__":
    main()
