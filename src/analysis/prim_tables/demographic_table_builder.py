from pathlib import Path
import pandas as pd
import numpy as np
from .effect_metrics import compute_stability, compute_pvalue, compute_cohens_d, interpret_effect_size
from src.utils.file_utils import load_csv_or_fail

BASELINE_SCENARIO = "NI"

def format_pvalue(p: float) -> str:
    """Formatta p-value con notazione scientifica se < 0.001"""
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return f"{p:.2e}"
    elif p < 0.01:
        return f"{p:.4f}"
    else:
        return f"{p:.3f}"

def build_demographic_table(csv_path: Path, summary_csv_path: Path, raw_csv_path: Path) -> pd.DataFrame:
    """
    Costruisce il DataFrame finale con metriche per la tabella demografica:
    scenario, coverage, density, SD(density), lift, effect_size_d, CI_lower/upper, p-value, stability, n_segment.
    
    FIXED: 
    - SD ora viene da prim_trajectory_summary.csv (density_std delle righe selezionate)
    - density ora viene da prim_trajectory_summary.csv (density_mean delle righe selezionate)
    - Gestisce correttamente scenari senza righe selezionate (es. NI)
    """
    df_base = load_csv_or_fail(csv_path)
    df_raw = load_csv_or_fail(raw_csv_path)
    df_summary = load_csv_or_fail(summary_csv_path)

    scenarios = df_raw["scenario"].unique()

    # 1️⃣ Estrai density_mean e density_std dalle righe selezionate in summary
    selected_summary = df_summary[df_summary["is_selected"] == True].set_index("scenario")
    
    # Per scenari senza righe selezionate, usa fallback da demographic_profiles
    density_map = {}
    density_sd_map = {}
    
    for s in scenarios:
        if s in selected_summary.index:
            # Usa i dati dalla trajectory summary (righe selezionate)
            density_map[s] = selected_summary.loc[s, "density_mean"]
            density_sd_map[s] = selected_summary.loc[s, "density_std"]
        else:
            # Fallback: usa demographic_profiles e SD = 0 (nessuna selezione)
            density_map[s] = df_base.loc[df_base["scenario"] == s, "density"].values[0]
            density_sd_map[s] = 0.0

    # 2️⃣ Cohen's d e CI
    effect_map = {
        s: dict(zip(["effect_size_d","effect_ci_lower","effect_ci_upper","effect_label"],
                    [*compute_cohens_d(df_raw, s, baseline=BASELINE_SCENARIO), 
                     interpret_effect_size(compute_cohens_d(df_raw, s, baseline=BASELINE_SCENARIO)[0])]))
        for s in scenarios
    }

    # 3️⃣ Costruzione records
    records = []
    for _, row in df_base.iterrows():
        s = row["scenario"]
        
        # Usa density dalla mappa (da summary se disponibile, altrimenti da base)
        density = density_map[s]
        density_sd = density_sd_map[s]
        
        coverage = row.get("coverage", np.nan)
        lift = row.get("lift", np.nan)
        n_segment = row.get("n_agents_segment", np.nan)
        stability = compute_stability(df_raw, s)
        
        # Per p_value, usa la density corretta del baseline
        baseline_density = density_map[BASELINE_SCENARIO]
        p_value = compute_pvalue(df_raw, baseline_density, s)

        rec = {
            "scenario": s,
            "coverage": coverage,
            "density": density,
            "density_sd": density_sd,
            "lift": lift,
            "effect_size_d": effect_map[s]["effect_size_d"],
            "effect_ci_lower": effect_map[s]["effect_ci_lower"],
            "effect_ci_upper": effect_map[s]["effect_ci_upper"],
            "p_value": format_pvalue(p_value),  # Formatta qui
            "stability": stability,
            "n_segment": n_segment
        }
        records.append(rec)

    return pd.DataFrame(records)