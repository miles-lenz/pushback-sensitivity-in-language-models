
import json
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


def save_probe_results(
    run_id: str,
    layer: int,
    act_type: str,
    y_test: t.Tensor,
    preds: t.Tensor,
    n_train: int,
    hyperparams: dict,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    dataset_name: str = "gsm8k",
    output_dir: str = "results/linear_probing",
) -> Path:
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    readable_date = now.strftime("%Y-%m-%d %H:%M:%S")

    y_test_np = y_test.cpu().numpy()
    preds_np = preds.cpu().numpy()

    # Metriken berechnen
    accuracy = float((preds == y_test).float().mean().item())
    baseline = float(
        max(
            (y_test == 1).float().mean().item(),
            (y_test == 0).float().mean().item(),
        )
    )
    clf_report = classification_report(
        y_test_np, preds_np, output_dict=True, zero_division=0
    )
    conf_matrix = confusion_matrix(y_test_np, preds_np).tolist()

    # Strukturierter Datensatz
    log_data = {
        "metadata": {
            "timestamp": readable_date,
            "run_id": run_id,
            "dataset": dataset_name,
            "model": model_name,
            "layer": layer,
            "activation_type": act_type,
            "token_position": "last_prompt_token",
        },
        "hyperparameters": hyperparams,
        "dataset_statistics": {
            "n_total": len(y_test) + n_train,
            "n_train": n_train,
            "n_test": len(y_test),
            "test_class_distribution": {
                "retained_0": int((y_test == 0).sum().item()),
                "changed_1": int((y_test == 1).sum().item()),
            },
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "majority_baseline": round(baseline, 4),
            "accuracy_above_baseline": round(accuracy - baseline, 4),
            "confusion_matrix": {
                "format": "[[TN, FP], [FN, TP]]",
                "matrix": conf_matrix,
            },
            "classification_report": clf_report,
        },
    }

    # Datei speichern
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filename = (
        save_path / f"{run_id}_{act_type}_layer{layer}_{timestamp_str}.json"
    )
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(f"\n[INFO] Ergebnisse erfolgreich gespeichert unter:\n-> {filename}")
    return filename


def compute_mass_mean_vector(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """Berechnet den Differenz-der-Mittelwerte-Vektor (Mass-Mean):

    v_mm = mean(x[y == 1]) - mean(x[y == 0])
    Richtung: von 'Retained (0)' hin zu 'Changed (1)'
    """
    mu_changed = x[y == 1].mean(dim=0)
    mu_retained = x[y == 0].mean(dim=0)
    return mu_changed - mu_retained


def compute_cosine_similarity(v1: t.Tensor, v2: t.Tensor) -> float:
    """Berechnet die Cosinus-Ähnlichkeit zwischen zwei 1D-Tensoren."""
    sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    return float(sim.item())


def save_cross_eval_results(
    run_id: str,
    layer: int,
    train_condition: str,
    eval_results: dict[str, dict],
    hyperparams: dict,
    probe_geometry: dict | None = None,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    dataset_name: str = "gsm8k",
    output_dir: str = "results/cross_evaluation",
) -> Path:
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    readable_date = now.strftime("%Y-%m-%d %H:%M:%S")

    log_data = {
        "metadata": {
            "timestamp": readable_date,
            "run_id": run_id,
            "dataset": dataset_name,
            "model": model_name,
            "layer": layer,
            "trained_on_condition": train_condition,
            "token_position": "last_prompt_token",
        },
        "hyperparameters": hyperparams,
        "probe_geometry": probe_geometry or {},
        "evaluations": {},
    }

    for cond_name, data in eval_results.items():
        y_true = data["y_true"]
        preds = data["preds"]
        probs = data["probs"]

        acc = float(accuracy_score(y_true, preds))
        bal_acc = float(balanced_accuracy_score(y_true, preds))
        baseline = float(max((y_true == 1).mean(), (y_true == 0).mean()))

        auc = None
        if len(set(y_true)) > 1:
            auc = float(roc_auc_score(y_true, probs))

        log_data["evaluations"][cond_name] = {
            "is_in_distribution": (cond_name == train_condition),
            "n_samples": len(y_true),
            "class_distribution": {
                "retained_0": int((y_true == 0).sum()),
                "changed_1": int((y_true == 1).sum()),
            },
            "metrics": {
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "roc_auc": round(auc, 4) if auc is not None else None,
                "majority_baseline": round(baseline, 4),
            },
            "confusion_matrix": {
                "format": "[[TN, FP], [FN, TP]]",
                "matrix": confusion_matrix(y_true, preds).tolist(),
            },
            "classification_report": classification_report(
                y_true, preds, output_dict=True, zero_division=0
            ),
        }

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        save_dir
        / f"{run_id}_train-{train_condition}_layer{layer}_{timestamp_str}.json"
    )
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(f"\n[INFO] Ergebnisse inkl. Geometrie gespeichert unter:\n-> {filename}")
    return filename