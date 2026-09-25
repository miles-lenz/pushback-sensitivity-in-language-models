import math
from pathlib import Path

import torch as t
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from linear_probing.data import (
    load_results,
)
from linear_probing.model import LRProbe

RUN_ID = "official_03"
LAYER = 22
raw_results = load_results(RUN_ID)

print("=" * 60)
print(f"ANALYSE: CONFIDENT WRONG & REVISION RATE (Layer {LAYER})")
print("=" * 60)


def get_confident_wrong_data(
    results: list[dict], pushback_type: str, layer: int
) -> tuple[t.Tensor, t.Tensor]:
    """Extrahiert Daten für 'Confident Wrong' (Beharrlich Falsch).

    Grundgesamtheit: Nur initial FALSCHE Aufgaben.
    Y = 1: Behält exakt dieselbe falsche Zahl bei (Confident Wrong / Stur).
    Y = 0: Ändert seine Zahl nach dem Pushback (Revidiert / Eingesehen).
    """
    x, y = [], []

    for res in results:
        init_ans = res.get("model_answer")
        ref_ans = res.get("reference_answer")
        pb_data = res.get("pushbacks", {}).get(pushback_type)

        if init_ans is None or ref_ans is None or pb_data is None:
            continue
        pb_ans = pb_data.get("model_answer")
        if pb_ans is None:
            continue

        # 1. FILTER: Nur initial falsche Aufgaben betrachten
        if math.isclose(init_ans, ref_ans, abs_tol=1e-4):
            continue  # War richtig -> ignorieren

        # 2. LABEL: Behält das Modell stur exakt dieselbe falsche Antwort?
        is_confident_wrong = 1 if math.isclose(init_ans, pb_ans, abs_tol=1e-4) else 0

        prompt_path = Path(pb_data["activations_prompt_path"])
        if not prompt_path.exists():
            continue

        acts = t.load(prompt_path, map_location="cpu", weights_only=True)
        x.append(acts[f"layer_{layer}"].float())
        y.append(is_confident_wrong)

    return t.stack(x), t.tensor(y, dtype=t.float)


def get_revision_rate_data(
    results: list[dict], pushback_type: str, layer: int
) -> tuple[t.Tensor, t.Tensor]:
    """Extrahiert Daten für die allgemeine 'Revision Rate' (Gesamte Umstimmungsbereitschaft).

    Grundgesamtheit: ALLE Aufgaben (richtig und falsch).
    Y = 1: Antwort wurde geändert (Revision).
    Y = 0: Antwort blieb unverändert (Persistenz).
    """
    x, y = [], []

    for res in results:
        init_ans = res.get("model_answer")
        pb_data = res.get("pushbacks", {}).get(pushback_type)

        if init_ans is None or pb_data is None:
            continue
        pb_ans = pb_data.get("model_answer")
        if pb_ans is None:
            continue

        # LABEL: Wurde die Zahl revidiert?
        has_revised = 0 if math.isclose(init_ans, pb_ans, abs_tol=1e-4) else 1

        prompt_path = Path(pb_data["activations_prompt_path"])
        if not prompt_path.exists():
            continue

        acts = t.load(prompt_path, map_location="cpu", weights_only=True)
        x.append(acts[f"layer_{layer}"].float())
        y.append(has_revised)

    return t.stack(x), t.tensor(y, dtype=t.float)


# --- 1. Verhaltens-Statistiken über den gesamten Datensatz ---
print("\n--- 1. Deskriptive Verhaltensraten ---")
for pb in ["adversarial", "medium", "weak"]:
    _, y_rev = get_revision_rate_data(raw_results, pb, LAYER)
    _, y_cw = get_confident_wrong_data(raw_results, pb, LAYER)

    rev_rate = (y_rev == 1).float().mean().item() * 100
    cw_rate = (y_cw == 1).float().mean().item() * 100

    print(f"[{pb.upper()}]")
    print(
        f"  - Allgemeine Revision Rate: {rev_rate:.2f}% ({int((y_rev == 1).sum())}/{len(y_rev)})"
    )
    print(
        f"  - Confident Wrong Rate:     {cw_rate:.2f}% ({int((y_cw == 1).sum())}/{len(y_cw)} der initial falschen)"
    )

# --- 2. Probe Training: Confident Wrong ---
print("\n--- 2. Probe Training: Confident Wrong (Train: Adversarial) ---")
x_cw, y_cw = get_confident_wrong_data(raw_results, "adversarial", LAYER)

indices = t.arange(len(x_cw))
idx_train, idx_test = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y_cw.numpy()
)

probe_cw = LRProbe.from_data(x_cw[idx_train], y_cw[idx_train], C=0.1)

# Mass-Mean & Cosinus-Ähnlichkeit
v_lr = probe_cw.direction.detach().cpu()
mu_cw = x_cw[idx_train][y_cw[idx_train] == 1].mean(dim=0)
mu_rev = x_cw[idx_train][y_cw[idx_train] == 0].mean(dim=0)
v_mm = mu_cw - mu_rev
cos_sim = F.cosine_similarity(v_lr.unsqueeze(0), v_mm.unsqueeze(0)).item()

print(f"Cosinus-Aehnlichkeit (v_LR vs. v_MM): {cos_sim:.4f}")
print(f"Norm(Mass-Mean): {v_mm.norm().item():.2f}")

for pb in ["adversarial", "medium", "weak"]:
    x_test_pb, y_test_pb = get_confident_wrong_data(raw_results, pb, LAYER)
    x_eval = x_test_pb[idx_test]
    y_eval = y_test_pb[idx_test].cpu().numpy()

    with t.no_grad():
        preds = probe_cw.pred(x_eval).cpu().numpy()
        probs = probe_cw(x_eval).cpu().numpy()

    bal_acc = balanced_accuracy_score(y_eval, preds)
    auc = roc_auc_score(y_eval, probs) if len(set(y_eval)) > 1 else float("nan")
    print(f"  Eval on {pb:<12} | Bal. Acc: {bal_acc * 100:5.2f}% | ROC-AUC: {auc:.4f}")
