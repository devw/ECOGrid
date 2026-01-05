from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

from .effect_metrics import compute_stability, compute_pvalue, compute_cohens_d, interpret_effect_size
from src.utils.file_utils import load_csv_or_fail

BASELINE_SCENARIO = "NI"


def build_demographic_table(
    csv_path: Path,
    summary_csv_path: Path,
    raw_csv_path: Path
) -> pd.DataFrame:
    """
    Costruisce il DataFrame finale con tutte le metriche richieste.
    
    Restituisce colonne interne numeriche:
    scenario, coverage, density, density_sd, lift,
    effect_size_d, effect_ci_lower, effect_ci_upper,
    p_value, stability, n_segment
    """
    df_summary = load_csv_or_fail(summary_csv_path)
    df_raw = load_csv_or_fail(raw_csv_path)
    df_base = load_csv_or_fail(csv_path)

    # Calcolo SD (density) per ogni scenario
    density_sd_map = {}
    for scenario in df_raw["scenario"].unique():
        adoption = df_raw.loc[df_raw["scenario"] == scenario, "is_selected"].astype(int).values

        p = adoption.mean()
        density_sd_map[scenario] = np.sqrt(p*(1-p))

    # CI e effect size
    effect_map = {}
    for scenario in df_raw["scenario"].unique():
        d, ci_lower, ci_upper = compute_cohens_d(df_raw, scenario, baseline=BASELINE_SCENARIO)
        effect_map[scenario] = {
            "effect_size_d": d,
            "effect_ci_lower": ci_lower,
            "effect_ci_upper": ci_upper,
            "effect_label": interpret_effect_size(d)
        }

    # Costruzione tabella
    records = []
    for _, row in df_base.iterrows():
        scenario = row["scenario"]
        density = row["density"]
        coverage = row.get("coverage", np.nan)
        lift = row.get("lift", np.nan)
        n_segment = row.get("n_agents_segment", np.nan)
        stability = compute_stability(df_raw, scenario)
        p_value = compute_pvalue(df_raw, df_raw.loc[df_raw["scenario"]==BASELINE_SCENARIO, "density"].values[0], scenario)

        rec = {
            "scenario": scenario,
            "coverage": coverage,
            "density": density,
            "density_sd": density_sd_map.get(scenario, np.nan),
            "lift": lift,
            "effect_size_d": effect_map[scenario]["effect_size_d"],
            "effect_ci_lower": effect_map[scenario]["effect_ci_lower"],
            "effect_ci_upper": effect_map[scenario]["effect_ci_upper"],
            "p_value": p_value,
            "stability": stability,
            "n_segment": n_segment
        }
        records.append(rec)

    df_final = pd.DataFrame(records)
    return df_final
