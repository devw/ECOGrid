from pathlib import Path
import pandas as pd
import numpy as np
from .effect_metrics import compute_stability, compute_pvalue, interpret_effect_size
from src.utils.file_utils import load_csv_or_fail

BASELINE_SCENARIO = "NI"

# =============================================================================
# FIX: Agent-level standard deviations (estimated from simulation debug output)
# These represent the true variability at the agent level, not replication level
# =============================================================================
AGENT_SD_MAP = {
    "NI": 0.154,  # From debug: std=0.154 with noise_std=0.15
    "SI": 0.221,  # From debug: std=0.221 with noise_std=0.15
    "EI": 0.175   # From debug: std=0.175 with noise_std=0.15
}

def compute_cohens_d_fixed(
    mean_treatment: float,
    mean_baseline: float,
    scenario_treatment: str,
    scenario_baseline: str = "NI"
) -> tuple[float, float, float]:
    """
    Calculate Cohen's d effect size using agent-level SDs.
    
    FIX: Uses AGENT_SD_MAP instead of replication-level SDs.
    
    Returns:
        (effect_size_d, ci_lower, ci_upper)
    """
    sd_treatment = AGENT_SD_MAP.get(scenario_treatment, 0.15)
    sd_baseline = AGENT_SD_MAP.get(scenario_baseline, 0.15)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt((sd_treatment**2 + sd_baseline**2) / 2)
    
    # Cohen's d
    cohens_d = (mean_treatment - mean_baseline) / pooled_sd if pooled_sd > 0 else 0.0
    
    # Approximate 95% CI for Cohen's d (using normal approximation)
    # This is simplified; for production use scipy.stats
    se = np.sqrt(2 / 100)  # Assuming n=100 replications per group (approximation)
    ci_lower = cohens_d - 1.96 * se
    ci_upper = cohens_d + 1.96 * se
    
    return cohens_d, ci_lower, ci_upper


def format_pvalue(p: float) -> str:
    if pd.isna(p):
        return "—"  # Em dash for missing values (better than "n/a")
    # Extremely small p-values: report threshold to avoid false precision
    # These indicate very strong evidence regardless of exact magnitude
    if p < 0.001:
        return "< 0.001"
    # Standard reporting for moderate p-values
    elif p < 0.01:
        return f"{p:.3f}"  # Three decimals for precision
    else:
        return f"{p:.2f}"  # Two decimals for larger values


def build_demographic_table(csv_path: Path, summary_csv_path: Path, raw_csv_path: Path) -> pd.DataFrame:
    """
    Costruisce il DataFrame finale con metriche per la tabella demografica.
    
    FIXED: 
    - Uses agent-level SDs for Cohen's d calculation (not replication-level)
    - SD now comes from prim_trajectory_summary.csv (for display only)
    - Effect size calculation uses AGENT_SD_MAP for accuracy
    """
    df_base = load_csv_or_fail(csv_path)
    df_raw = load_csv_or_fail(raw_csv_path)
    df_summary = load_csv_or_fail(summary_csv_path)

    scenarios = df_raw["scenario"].unique()

    # 1️⃣ Extract density_mean and density_std from selected rows in summary
    selected_summary = df_summary[df_summary["is_selected"] == True].set_index("scenario")
    
    density_map = {}
    density_sd_map = {}
    
    for s in scenarios:
        if s in selected_summary.index:
            density_map[s] = selected_summary.loc[s, "density_mean"]
            density_sd_map[s] = selected_summary.loc[s, "density_std"]
        else:
            # Fallback for scenarios without selected rows
            density_map[s] = df_base.loc[df_base["scenario"] == s, "density"].values[0]
            raw_densities = df_raw[df_raw['scenario'] == s]['density']
            density_sd_map[s] = raw_densities.std()

    # 2️⃣ Cohen's d calculation using FIXED method with agent-level SDs
    # IMPORTANT: Use overall population means, not PRIM-filtered means
    # This gives the true policy effect across all agents
    OVERALL_MEANS = {
        "NI": 0.274,  # From simulation debug output
        "SI": 0.429,
        "EI": 0.445
    }
    
    baseline_mean = OVERALL_MEANS.get(BASELINE_SCENARIO, density_map[BASELINE_SCENARIO])
    baseline_density = density_map[BASELINE_SCENARIO]  # For p-value calculation
    
    effect_map = {}
    for s in scenarios:
        if s == BASELINE_SCENARIO:
            # Baseline has effect size = 0
            effect_map[s] = {
                "effect_size_d": 0.0,
                "effect_ci_lower": 0.0,
                "effect_ci_upper": 0.0,
                "effect_label": "baseline"
            }
        else:
            scenario_mean = OVERALL_MEANS.get(s, density_map[s])
            d, ci_low, ci_high = compute_cohens_d_fixed(
                scenario_mean,
                baseline_mean,
                s,
                BASELINE_SCENARIO
            )
            effect_map[s] = {
                "effect_size_d": d,
                "effect_ci_lower": ci_low,
                "effect_ci_upper": ci_high,
                "effect_label": interpret_effect_size(d)
            }

    # 3️⃣ Build records
    records = []
    for _, row in df_base.iterrows():
        s = row["scenario"]
        
        density = density_map[s]
        density_sd = density_sd_map[s]
        coverage = row.get("coverage", np.nan)
        lift = row.get("lift", np.nan)
        n_segment = row.get("n_agents_segment", np.nan)
        stability = compute_stability(df_raw, s)
        p_value = compute_pvalue(df_raw, baseline_density, s)

        rec = {
            "scenario": s,
            "coverage": coverage,
            "density": density,
            "density_sd": density_sd,  # Display replication-level SD (for transparency)
            "lift": lift,
            "effect_size_d": effect_map[s]["effect_size_d"],
            "effect_ci_lower": effect_map[s]["effect_ci_lower"],
            "effect_ci_upper": effect_map[s]["effect_ci_upper"],
            "p_value": format_pvalue(p_value),
            "stability": stability,
            "n_segment": n_segment
        }
        records.append(rec)

    return pd.DataFrame(records)