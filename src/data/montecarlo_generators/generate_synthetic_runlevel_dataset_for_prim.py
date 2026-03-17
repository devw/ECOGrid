import numpy as np
import pandas as pd
from pathlib import Path

# ==============================
# CONFIG
# ==============================

N_RUNS = 100
SCENARIOS = ["NI", "EI", "SI"]
OUTPUT_PATH = "data/run_level_dataset.csv"

np.random.seed(42)


# ==============================
# GENERATION LOGIC
# ==============================

def generate_base_features(n):
    """
    Genera struttura socio-economica di base.
    """
    income = np.random.beta(2, 2, n)          # distribuzione realistica [0,1]
    trust = np.random.beta(2, 2, n)
    connectivity = np.random.normal(0.5, 0.1, n)
    connectivity = np.clip(connectivity, 0, 1)

    return income, trust, connectivity


def adoption_NI(income, trust, connectivity):
    """
    No incentive:
    Debole dipendenza strutturale.
    """
    noise = np.random.normal(0, 0.04, len(income))
    adoption = 0.28 + 0.05 * income + 0.03 * trust + noise
    return np.clip(adoption, 0, 1)


def adoption_EI(income, trust, connectivity):
    """
    Economic incentive:
    Effetto diffuso e lineare.
    """
    noise = np.random.normal(0, 0.03, len(income))
    adoption = 0.30 + 0.25 * income + 0.20 * trust + noise
    return np.clip(adoption, 0, 1)


def adoption_SI(income, trust, connectivity):
    """
    Service incentive:
    Effetto soglia (concentrazione su high income + high trust).
    """
    noise = np.random.normal(0, 0.03, len(income))

    threshold_effect = ((income > 0.6) & (trust > 0.6)).astype(float)

    adoption = 0.28 + 0.35 * threshold_effect + 0.05 * income + noise
    return np.clip(adoption, 0, 1)


# ==============================
# MAIN
# ==============================

def main():

    all_rows = []

    for scenario in SCENARIOS:

        income, trust, connectivity = generate_base_features(N_RUNS)

        if scenario == "NI":
            adoption = adoption_NI(income, trust, connectivity)

        elif scenario == "EI":
            adoption = adoption_EI(income, trust, connectivity)

        elif scenario == "SI":
            adoption = adoption_SI(income, trust, connectivity)

        for i in range(N_RUNS):
            all_rows.append({
                "run": i,
                "scenario": scenario,
                "final_adoption": adoption[i],
                "mean_income": income[i],
                "mean_trust": trust[i],
                "mean_connectivity": connectivity[i]
            })

    df = pd.DataFrame(all_rows)

    Path("data").mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("✔ Synthetic dataset generated")
    print(f"✔ Total rows: {len(df)}")
    print(f"✔ Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()