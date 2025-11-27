from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

# Importa SCENARIOS dai settings, necessario per l'iterazione
from .._config.settings import SCENARIOS, P_VALUE_THRESHOLD 

# --- STATISTICAL ANALYSIS ---

def compute_pairwise_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Welch t-test p-values per grid cell between scenarios."""
    rows = []
    for (t, y), group in df.groupby(["trust_bin", "income_bin"]):
        for s1, s2 in combinations(SCENARIOS.keys(), 2):
            a = group[group["scenario"] == s1]
            b = group[group["scenario"] == s2]
            if a.empty or b.empty or len(a) == 0 or len(b) == 0:
                continue

            mean1, std1, n1 = a[["adoption_rate", "std_dev", "n_replications"]].values[0]
            mean2, std2, n2 = b[["adoption_rate", "std_dev", "n_replications"]].values[0]

            se2 = (std1**2 / n1) + (std2**2 / n2)
            if se2 <= 0:
                pval = 1.0
            else:
                tstat = (mean1 - mean2) / np.sqrt(se2)
                denom = (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1)
                # Degrees of freedom calculation (Welch–Satterthwaite equation simplified)
                df_ = (se2 ** 2) / denom if denom > 0 else (n1 + n2 - 2)
                pval = 2 * (1 - stats.t.cdf(abs(tstat), df_))

            rows.append({
                "trust_bin": t,
                "income_bin": y,
                "scenario_A": s1,
                "scenario_B": s2,
                "mean_diff": mean1 - mean2,
                "p_value": pval,
                "significant": pval < P_VALUE_THRESHOLD,
            })

    return pd.DataFrame(rows)