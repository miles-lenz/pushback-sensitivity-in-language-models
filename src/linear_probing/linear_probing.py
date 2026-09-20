"""
The goal of this script is to find out if we can predict that the answer
of the model will change or not based on the linear probing results.

We make it simple.
 - Run: results_03
 - Pushback: (adversial)
 - Layer: 14
 - Token: Last pushback token before llm answer
 - Train/Test: 80/20
"""

import torch as t
from sklearn.model_selection import train_test_split

from data import get_activations, load_results
from model import LRProbe
from utils import mass_mean_vector, save_probe_results


def run_probe_experiment(
    x: t.Tensor,
    y: t.Tensor,
    run_id: str,
    layer: int,
    act_type: str,
    test_size: float = 0.2,
    C: float = 0.1,
):
    # 1. Split
    indices = t.arange(len(x))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=y.numpy()
    )

    x_train, y_train = x[idx_train], y[idx_train]
    x_test, y_test = x[idx_test], y[idx_test]

    # 2. Probe fitten & vorhersagen
    probe = LRProbe.from_data(x_train, y_train, C=C)
    with t.no_grad():
        preds = probe.pred(x_test)

    # 3. Ergebnisse automatisch in Datei abspeichern
    hyperparams = {
        "C": C,
        "test_size": test_size,
        "random_state": 42,
        "scaler": "StandardScaler",
    }
    save_probe_results(
        run_id=run_id,
        layer=layer,
        act_type=act_type,
        y_test=y_test,
        preds=preds,
        n_train=len(x_train),
        hyperparams=hyperparams,
    )


if __name__ == "__main__":
    RUN_ID = "official_03"
    LAYER = 14
    ACT_TYPE = "adversarial_prompt"

    res = load_results(RUN_ID)
    x, y = get_activations(res, act_type1=ACT_TYPE, layer=LAYER)

    mass_mean_vector(x, y)

    run_probe_experiment(x=x, y=y, run_id=RUN_ID, layer=LAYER, act_type=ACT_TYPE)
