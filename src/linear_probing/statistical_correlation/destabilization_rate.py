import json
import math
from datetime import datetime
from pathlib import Path

import torch as t
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from linear_probing.data import (
    extract_activations_and_labels,
    get_initially_correct_tasks,
    load_results,
)
from linear_probing.model import LRProbe
from linear_probing.utils import compute_mass_mean_vector

# ==============================================================================
# Configuration
# ==============================================================================
RUN_ID = "official_03"
LAYER = 27  # Optimal decision layer identified in prior experiments
TRAIN_ON = "adversarial"
TEST_SIZE = 0.2
C_REG = 0.1
RANDOM_STATE = 42


print("--- Linear Probing Destabilization Evaluation ---")


def save_destabilization_run(
    run_id: str,
    layer: int,
    train_condition: str,
    eval_results: dict,
    hyperparams: dict,
    geometry: dict,
    output_dir: str = "results/destabilization",
) -> Path:
    now = datetime.now()  # noqa: DTZ005
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    log_data = {
        "metadata": {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "target": "destabilization_rate (Right -> Wrong)",
            "layer": layer,
            "trained_on_condition": train_condition,
            "token_position": "last_prompt_token",
        },
        "hyperparameters": hyperparams,
        "probe_geometry": geometry,
        "evaluations": eval_results,
    }

    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filename = (
        save_path
        / f"{run_id}_destabilization_train-{train_condition}_layer{layer}_{timestamp}.json"
    )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(f"\n[INFO] Run results saved to:\n-> {filename}")
    return filename


def main():
    print(f"Loading data for run: {RUN_ID} ...")
    raw_results = load_results(RUN_ID)

    # 1. Problem-level filter: Keep only initially correct tasks
    correct_tasks = get_initially_correct_tasks(raw_results)
    print(
        f"Dataset summary: {len(correct_tasks)} / {len(raw_results)} problems initially answered correctly."
    )

    # 2. Extract activations for all 3 pushback conditions on the SAME tasks
    print(f"Extracting Layer {LAYER} activations...")
    data = {}
    for pb_type in ["adversarial", "medium", "weak"]:
        x, y = extract_activations_and_labels(correct_tasks, pb_type, LAYER)
        data[pb_type] = (x, y)
        n_caved = int((y == 1).sum().item())
        print(
            f"  - {pb_type:<12}: {n_caved}/{len(y)} destabilized ({n_caved / len(y) * 100:.2f}%)"
        )

    # 3. Stratified Train/Test split based on the training condition
    x_train_source, y_train_source = data[TRAIN_ON]
    indices = t.arange(len(x_train_source))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_source.numpy(),
    )

    # 4. Train Probe (LRProbe with StandardScaler & fit_intercept=True)
    print(f"\nFitting LRProbe on {TRAIN_ON} (train samples: {len(idx_train)}) ...")
    probe = LRProbe.from_data(
        x_train_source[idx_train], y_train_source[idx_train], C=C_REG
    )

    # 5. Geometrical Analysis (Mass-Mean vs. LR Direction)
    v_lr = probe.direction.detach().cpu()
    v_mm = compute_mass_mean_vector(
        x_train_source[idx_train], y_train_source[idx_train]
    )
    cos_sim = float(F.cosine_similarity(v_lr.unsqueeze(0), v_mm.unsqueeze(0)).item())

    geometry_info = {
        "cosine_similarity_lr_vs_mm": round(cos_sim, 4),
        "norm_mass_mean": round(float(v_mm.norm().item()), 4),
        "norm_lr_direction": round(float(v_lr.norm().item()), 4),
    }

    print("\n--- Geometry (Training Set) ---")
    print(f"Cosine Similarity (v_LR, v_MM): {cos_sim:.4f}")
    print(f"Norm(v_MM): {geometry_info['norm_mass_mean']:.2f}")

    # 6. Cross-Evaluation on Held-Out Test Split
    eval_results = {}
    print(
        f"\n{'Condition':<25} | {'Acc':<8} | {'Bal Acc':<10} | {'ROC-AUC':<8} | {'Baseline':<8}"
    )
    print("-" * 68)

    for cond_name, (x_full, y_full) in data.items():
        x_test, y_test = x_full[idx_test], y_full[idx_test]

        with t.no_grad():
            preds = probe.pred(x_test).cpu().numpy()
            probs = probe(x_test).cpu().numpy()

        y_true = y_test.cpu().numpy()

        acc = float(accuracy_score(y_true, preds))
        bal_acc = float(balanced_accuracy_score(y_true, preds))
        auc = (
            float(roc_auc_score(y_true, probs))
            if len(set(y_true)) > 1
            else float("nan")
        )
        baseline = float(max((y_true == 1).mean(), (y_true == 0).mean()))

        eval_results[cond_name] = {
            "is_in_distribution": (cond_name == TRAIN_ON),
            "n_samples": len(y_true),
            "class_distribution": {
                "robust_0": int((y_true == 0).sum()),
                "destabilized_1": int((y_true == 1).sum()),
            },
            "metrics": {
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "roc_auc": round(auc, 4) if not math.isnan(auc) else None,
                "majority_baseline": round(baseline, 4),
            },
            "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
            "classification_report": classification_report(
                y_true, preds, output_dict=True, zero_division=0
            ),
        }

        auc_str = f"{auc:.4f}" if not math.isnan(auc) else "N/A"
        print(
            f"{cond_name:<25} | {acc * 100:6.2f}% | {bal_acc * 100:8.2f}% | {auc_str:<8} | {baseline * 100:6.2f}%"
        )

    # 7. Save output
    hyperparams = {
        "C": C_REG,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "scaler": "StandardScaler",
        "fit_intercept": True,
    }

    save_destabilization_run(
        run_id=RUN_ID,
        layer=LAYER,
        train_condition=TRAIN_ON,
        eval_results=eval_results,
        hyperparams=hyperparams,
        geometry=geometry_info,
    )


if __name__ == "__main__":
    main()
