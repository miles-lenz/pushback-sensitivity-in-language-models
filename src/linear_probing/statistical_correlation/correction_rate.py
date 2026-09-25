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
    get_initially_incorrect_tasks,
    load_results,
)
from linear_probing.model import LRProbe

# ==============================================================================
# Konfiguration
# ==============================================================================
RUN_ID = "official_03"
LAYER = 27  # Oder 14 / 27
TRAIN_ON = "adversarial"
TEST_SIZE = 0.2
C_REG = 0.1
RANDOM_STATE = 42


# ==============================================================================
# Datenfilterung & Extraktion
# ==============================================================================
def extract_correction_data(
    tasks: list[dict], pushback_type: str, layer: int
) -> tuple[t.Tensor, t.Tensor]:
    """Extrahiert Aktivierungen (X) und Korrektur-Labels (Y).

    Y = 0: Bleibt falsch (stur/falsch geändert)
    Y = 1: Korrigiert zur korrekten Referenzantwort
    """
    x, y = [], []

    for task in tasks:
        ref_ans = task["reference_answer"]
        pb_data = task.get("pushbacks", {}).get(pushback_type)
        if pb_data is None or pb_data.get("model_answer") is None:
            continue

        pb_ans = pb_data["model_answer"]
        # Hat das Modell jetzt die richtige Antwort gefunden?
        is_corrected = 1 if math.isclose(pb_ans, ref_ans, abs_tol=1e-4) else 0

        prompt_path = Path(pb_data["activations_prompt_path"])
        if not prompt_path.exists():
            continue

        acts = t.load(prompt_path, map_location="cpu", weights_only=True)
        x.append(acts[f"layer_{layer}"].float())
        y.append(is_corrected)

    return t.stack(x), t.tensor(y, dtype=t.float)


def compute_mass_mean_vector(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """v_mm = mean(corrected) - mean(remained_incorrect)"""
    mu_corrected = x[y == 1].mean(dim=0)
    mu_incorrect = x[y == 0].mean(dim=0)
    return mu_corrected - mu_incorrect


def save_correction_run(
    run_id: str,
    layer: int,
    train_condition: str,
    eval_results: dict,
    hyperparams: dict,
    geometry: dict,
    output_dir: str = "results/correction_rate",
) -> Path:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    log_data = {
        "metadata": {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "target": "correction_rate (Wrong -> Right)",
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
        / f"{run_id}_correction_train-{train_condition}_layer{layer}_{timestamp}.json"
    )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(f"\n[INFO] Ergebnisse gespeichert unter:\n-> {filename}")
    return filename


# ==============================================================================
# Main
# ==============================================================================
def main():
    print(f"Lade Daten für Run: {RUN_ID} ...")
    raw_results = load_results(RUN_ID)

    # 1. Nur initial falsche Aufgaben isolieren
    incorrect_tasks = get_initially_incorrect_tasks(raw_results)
    print(
        f"Gefiltert: {len(incorrect_tasks)} / {len(raw_results)} Aufgaben waren initial FALSCH."
    )

    # 2. Daten für alle Pushbacks extrahieren & Baseline prüfen
    print(f"\nExtrahiere Aktivierungen für Layer {LAYER}...")
    data = {}
    for pb_type in ["adversarial", "medium", "weak"]:
        x, y = extract_correction_data(incorrect_tasks, pb_type, LAYER)
        data[pb_type] = (x, y)
        n_corrected = int((y == 1).sum().item())
        n_total = len(y)
        corr_rate = (n_corrected / n_total) * 100 if n_total > 0 else 0.0
        print(
            f"  - {pb_type:<12}: {n_corrected}/{n_total} korrigiert ({corr_rate:.2f}%)"
        )

    # 3. Train/Test-Split auf Basis der Trainings-Bedingung
    x_train_src, y_train_src = data[TRAIN_ON]

    # Sicherheitsprüfung: Wurde überhaupt etwas korrigiert?
    if (y_train_src == 1).sum() < 2:
        print(
            f"\n[ABBRUCH] Zu wenige positive Korrektur-Fälle in {TRAIN_ON} für ein sinnvolles Training!"
        )
        return

    indices = t.arange(len(x_train_src))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_src.numpy(),
    )

    # 4. Probe fitten
    print(
        f"\nTrainiere LRProbe auf: {TRAIN_ON} ({len(idx_train)} Trainingsbeispiele)..."
    )
    probe = LRProbe.from_data(x_train_src[idx_train], y_train_src[idx_train], C=C_REG)

    # 5. Geometrie (Mass-Mean vs LR)
    v_lr = probe.direction.detach().cpu()
    v_mm = compute_mass_mean_vector(x_train_src[idx_train], y_train_src[idx_train])
    cos_sim = float(F.cosine_similarity(v_lr.unsqueeze(0), v_mm.unsqueeze(0)).item())

    geometry_info = {
        "cosine_similarity_lr_vs_mm": round(cos_sim, 4),
        "norm_mass_mean": round(float(v_mm.norm().item()), 4),
        "norm_lr_direction": round(float(v_lr.norm().item()), 4),
    }

    print("\n--- Geometrie (Trainingsdaten) ---")
    print(f"Cosinus-Aehnlichkeit (v_LR, v_MM): {cos_sim:.4f}")
    print(f"Norm Mass-Mean: {geometry_info['norm_mass_mean']:.2f}")

    # 6. Evaluation
    eval_results = {}
    print(
        f"\n{'Bedingung':<25} | {'Acc':<8} | {'Bal Acc':<10} | {'ROC-AUC':<8} | {'Baseline':<8}"
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
                "remained_incorrect_0": int((y_true == 0).sum()),
                "corrected_1": int((y_true == 1).sum()),
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

    # 7. Speichern
    save_correction_run(
        run_id=RUN_ID,
        layer=LAYER,
        train_condition=TRAIN_ON,
        eval_results=eval_results,
        hyperparams={
            "C": C_REG,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "scaler": "StandardScaler",
            "fit_intercept": True,
        },
        geometry=geometry_info,
    )


if __name__ == "__main__":
    main()
