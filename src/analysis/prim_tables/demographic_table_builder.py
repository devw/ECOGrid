from pathlib import Path
import pandas as pd
import numpy as np
from .effect_metrics import compute_stability, compute_pvalue, compute_cohens_d, interpret_effect_size
from src.utils.file_utils import load_csv_or_fail

BASELINE_SCENARIO = "NI"

def build_demographic_table(csv_path: Path, summary_csv_path: Path, raw_csv_path: Path) -> pd.DataFrame:
    """
    Costruisce il DataFrame finale con metriche per la tabella demografica:
    scenario, coverage, density, SD(density), lift, effect_size_d, CI_lower/upper, p-value, stability, n_segment.
    SD calcolato come SD per variabile binaria: sqrt(p*(1-p)).
    """
    df_base = load_csv_or_fail(csv_path)
    df_raw = load_csv_or_fail(raw_csv_path)
    df_summary = load_csv_or_fail(summary_csv_path)

    scenarios = df_raw["scenario"].unique()

    # 1️⃣ SD binaria per scenario
    density_sd_map = {s: np.sqrt(df_raw.loc[df_raw["scenario"] == s, "is_selected"].astype(int).mean()
                                  * (1 - df_raw.loc[df_raw["scenario"] == s, "is_selected"].astype(int).mean()))
                      for s in scenarios}

    # 2️⃣ Cohen's d e CI
    effect_map = {
        s: dict(zip(["effect_size_d","effect_ci_lower","effect_ci_upper","effect_label"],
                    [*compute_cohens_d(df_raw, s, baseline=BASELINE_SCENARIO), interpret_effect_size(compute_cohens_d(df_raw, s, baseline=BASELINE_SCENARIO)[0])]))
        for s in scenarios
    }

    # 3️⃣ Costruzione records
    records = []
    for _, row in df_base.iterrows():
        s = row["scenario"]
        density = row["density"]
        coverage = row.get("coverage", np.nan)
        lift = row.get("lift", np.nan)
        n_segment = row.get("n_agents_segment", np.nan)
        stability = compute_stability(df_raw, s)
        p_value = compute_pvalue(df_raw, df_raw.loc[df_raw["scenario"]==BASELINE_SCENARIO, "density"].values[0], s)

        rec = {
            "scenario": s,
            "coverage": coverage,
            "density": density,
            "density_sd": density_sd_map.get(s, np.nan),
            "lift": lift,
            "effect_size_d": effect_map[s]["effect_size_d"],
            "effect_ci_lower": effect_map[s]["effect_ci_lower"],
            "effect_ci_upper": effect_map[s]["effect_ci_upper"],
            "p_value": p_value,
            "stability": stability,
            "n_segment": n_segment
        }
        records.append(rec)

    return pd.DataFrame(records)
