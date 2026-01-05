from typing import Optional
import pandas as pd
import numpy as np

# ----------------------------------------------------
# Metriche statistiche base
# ----------------------------------------------------

def compute_stability(raw_df: pd.DataFrame, scenario: str) -> float:
    """
    Calcola la stabilità del segmento: proporzione di repliche in cui il segmento è selezionato.
    
    raw_df: DataFrame con colonne ["scenario", "is_selected"]
    scenario: nome dello scenario da valutare
    """
    return raw_df.loc[raw_df["scenario"] == scenario, "is_selected"].mean()


def compute_pvalue(raw_df: pd.DataFrame, baseline_density: float, scenario: str) -> float:
    """
    Calcola il p-value empirico confrontando la densità dello scenario con il baseline.
    
    raw_df: DataFrame con colonne ["scenario", "density"]
    baseline_density: densità media del baseline scenario
    scenario: scenario da valutare
    """
    densities = raw_df.loc[raw_df["scenario"] == scenario, "density"].values
    return max((densities <= baseline_density).mean(), 1e-4)


def compute_cohens_d(
    raw_df: pd.DataFrame,
    scenario: str,
    baseline: str = "NI",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calcola Cohen's d e CI bootstrap tra scenario e baseline.
    
    Restituisce: (d, ci_lower, ci_upper)
    Se baseline assente o pooled_sd = 0 => (None, None, None)
    
    raw_df: DataFrame con colonne ["scenario", "density"]
    scenario: scenario da confrontare
    baseline: scenario baseline (default "NI")
    n_bootstrap: numero di replicates per bootstrap CI
    ci_level: livello di confidenza (default 0.95)
    """
    base = raw_df.loc[raw_df["scenario"] == baseline, "density"].values
    comp = raw_df.loc[raw_df["scenario"] == scenario, "density"].values
    if len(base) == 0 or len(comp) == 0:
        return None, None, None

    # pooled sd
    pooled_sd = np.sqrt((np.std(base, ddof=1)**2 + np.std(comp, ddof=1)**2)/2)
    if pooled_sd == 0:
        return None, None, None

    d = (np.mean(comp) - np.mean(base)) / pooled_sd

    # bootstrap CI
    boot_d = []
    rng = np.random.default_rng()
    for _ in range(n_bootstrap):
        sample_base = rng.choice(base, size=len(base), replace=True)
        sample_comp = rng.choice(comp, size=len(comp), replace=True)
        pooled_sd_b = np.sqrt((np.std(sample_base, ddof=1)**2 + np.std(sample_comp, ddof=1)**2)/2)
        if pooled_sd_b == 0:
            continue
        boot_d.append((np.mean(sample_comp) - np.mean(sample_base))/pooled_sd_b)

    if len(boot_d) == 0:
        ci_lower, ci_upper = None, None
    else:
        alpha = 1 - ci_level
        ci_lower = np.percentile(boot_d, 100*alpha/2)
        ci_upper = np.percentile(boot_d, 100*(1-alpha/2))

    return d, ci_lower, ci_upper


def interpret_effect_size(d: Optional[float]) -> str:
    """
    Interpreta l'effetto secondo convenzioni Cohen:
    negligible (<0.2), small (<0.5), medium (<0.8), large (>=0.8)
    """
    if d is None:
        return "n/a"
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"
