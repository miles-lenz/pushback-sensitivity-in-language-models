"""! TODO: Refactor
This is basically the same as lr_prob.py but we train on one pushback type and evaluate on all three.
"""

import torch as t
from sklearn.model_selection import train_test_split

from linear_probing.data import get_activations, load_results
from linear_probing.model import LRProbe
from linear_probing.utils import (
    compute_cosine_similarity,
    compute_mass_mean_vector,
    save_cross_eval_results,
)

print("--- Linear Probing Cross Evaluation ---")
RUN_ID = "official_03"
LAYER = 22
TRAIN_ON = "adversarial"  # "adversarial", "medium", "weak"
TEST_SIZE = 0.2
C_REG = 0.1

res = load_results(RUN_ID)

# Daten laden
x_adv, y_adv = get_activations(res, "adversarial", layer=LAYER)
x_med, y_med = get_activations(res, "medium", layer=LAYER)
x_weak, y_weak = get_activations(res, "weak", layer=LAYER)

data_dict = {
    "adversarial": (x_adv, y_adv),
    "medium": (x_med, y_med),
    "weak": (x_weak, y_weak),
}

# Split
x_train_src, y_train_src = data_dict[TRAIN_ON]
indices = t.arange(len(x_train_src))
idx_train, idx_test = train_test_split(
    indices, test_size=TEST_SIZE, random_state=42, stratify=y_train_src.numpy()
)

# LR-Probe trainieren
probe = LRProbe.from_data(x_train_src[idx_train], y_train_src[idx_train], C=C_REG)
v_lr = probe.direction.detach().cpu()

# Mass Mean Vektor berechnen
v_mm = compute_mass_mean_vector(x_train_src[idx_train], y_train_src[idx_train])

# Cos sim
cos_sim = compute_cosine_similarity(v_lr, v_mm)
print(f"--- Geometrische Analyse (Trainingsdaten: {TRAIN_ON}) ---")
print(f"Cosinus-Aehnlichkeit (LR-Richtung vs. Mass-Mean): {cos_sim:.4f}")
print(f"Norm(v_mm): {v_mm.norm().item():.2f} | Norm(v_lr): {v_lr.norm().item():.2f}\n")

# Evaluation
eval_results = {}
for cond_name, (x_full, y_full) in data_dict.items():
    x_test = x_full[idx_test]
    y_test = y_full[idx_test]

    with t.no_grad():
        preds = probe.pred(x_test).cpu().numpy()
        probs = probe(x_test).cpu().numpy()

    eval_results[cond_name] = {
        "y_true": y_test.cpu().numpy(),
        "preds": preds,
        "probs": probs,
    }

# Save
probe_geometry_info = {
    "cosine_similarity_lr_vs_mm": round(cos_sim, 4),
    "norm_mass_mean": round(float(v_mm.norm().item()), 4),
    "norm_lr_direction": round(float(v_lr.norm().item()), 4),
    "interpretation": (
        "High alignment (>0.70) indicates clean linear geometry; "
        "low alignment indicates LR is heavily relying on boundary noise."
    ),
}

save_cross_eval_results(
    run_id=RUN_ID,
    layer=LAYER,
    train_condition=TRAIN_ON,
    eval_results=eval_results,
    hyperparams={"C": C_REG, "test_size": TEST_SIZE, "scaler": "StandardScaler"},
    probe_geometry=probe_geometry_info,
)
