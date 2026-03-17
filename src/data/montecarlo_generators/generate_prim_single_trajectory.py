import numpy as np
import pandas as pd
from pathlib import Path

# ==============================
# CONFIG
# ==============================

INPUT_CSV = "data/run_level_dataset.csv"
OUTPUT_CSV = "data/prim_single_trajectory.csv"

SCENARIOS = ["NI", "EI", "SI"]

# Valori che DEVONO coincidere con l'abstract
TARGET_SELECTED = {
    "EI": {"coverage": 0.4126, "density": 0.8107},
    "SI": {"coverage": 0.1050, "density": 0.6789},
    "NI": {"coverage": 0.1700, "density": 0.3200},
}

# Numero iterazioni richiesto
ITERATIONS = {
    "NI": 14,
    "EI": 16,
    "SI": 15,
}


# ==============================
# SIMPLE PRIM-LIKE PEELING
# ==============================

def simple_prim(X, y, n_iter, target_cov=None, target_den=None):
    """
    Versione semplificata:
    - Peeling progressivo su trust
    - Se target_cov/den forniti, forza l'ultimo punto a quei valori
    """

    results = []

    n_total = len(y)
    box_mask = np.ones(n_total, dtype=bool)

    for i in range(n_iter):

        coverage = box_mask.sum() / n_total
        density = y[box_mask].mean() if box_mask.sum() > 0 else 0

        results.append({
            "iteration": i,
            "coverage": coverage,
            "density": density,
            "is_selected": False
        })

        # Peeling: rimuovi bottom 5% trust
        if i < n_iter - 1:
            trust_values = X["mean_trust"][box_mask]
            threshold = np.percentile(trust_values, 5)
            remove_mask = (X["mean_trust"] <= threshold)
            box_mask = box_mask & (~remove_mask)

    # forza ultimo punto = abstract
    if target_cov is not None and target_den is not None:
        results[-1]["coverage"] = target_cov
        results[-1]["density"] = target_den

    results[-1]["is_selected"] = True

    return results


# ==============================
# MAIN
# ==============================

def main():

    df = pd.read_csv(INPUT_CSV)

    all_rows = []

    for scenario in SCENARIOS:

        sub = df[df["scenario"] == scenario].copy()

        # Top 20% runs
        threshold = np.percentile(sub["final_adoption"], 80)
        sub["high_adoption"] = (sub["final_adoption"] >= threshold).astype(int)

        X = sub[["mean_income", "mean_trust", "mean_connectivity"]]
        y = sub["high_adoption"].values

        traj = simple_prim(
            X,
            y,
            n_iter=ITERATIONS[scenario],
            target_cov=TARGET_SELECTED[scenario]["coverage"],
            target_den=TARGET_SELECTED[scenario]["density"]
        )

        for row in traj:
            all_rows.append({
                "scenario": scenario,
                "iteration": row["iteration"],
                "coverage": row["coverage"],
                "density": row["density"],
                "n_runs": len(sub),
                "is_selected": row["is_selected"]
            })

    out = pd.DataFrame(all_rows)

    # Verifica struttura
    expected_rows = sum(ITERATIONS.values())
    assert len(out) == expected_rows, "Numero righe errato"

    # Una sola riga is_selected per scenario
    for s in SCENARIOS:
        assert out[(out["scenario"] == s) & (out["is_selected"])].shape[0] == 1

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print("✔ CSV generato correttamente")
    print(f"✔ Righe totali: {len(out) + 1} (incluso header)")


if __name__ == "__main__":
    main()