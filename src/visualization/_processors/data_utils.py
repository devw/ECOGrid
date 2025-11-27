import numpy as np
import pandas as pd

# --- GRID / PIVOT HELPERS ---

def _pivot_grid(df: pd.DataFrame, value_col: str) -> np.ndarray:
    """
    Pivot the DataFrame and return a 2D numpy array aligned by income_bin (rows) and trust_bin (cols).
    """
    table = (
        df.pivot_table(index="income_bin", columns="trust_bin", values=value_col, fill_value=0)
          .sort_index()
          .sort_index(axis=1)
    )
    return table.values


def scenario_grid(df: pd.DataFrame, scenario: str) -> dict:
    """
    Extract grid arrays (adoption rate, std dev, CI) and basic metadata for a specific scenario.
    """
    sub = df[df["scenario"] == scenario].copy()
    if sub.empty:
        raise ValueError(f"No data for scenario {scenario}")

    trust = np.sort(sub["trust_bin"].unique())
    income = np.sort(sub["income_bin"].unique())

    return {
        "trust": trust,
        "income": income,
        "adoption": _pivot_grid(sub, "adoption_rate"),
        "std_dev": _pivot_grid(sub, "std_dev"),
        "ci_lower": _pivot_grid(sub, "ci_lower"),
        "ci_upper": _pivot_grid(sub, "ci_upper"),
        "n_replications": int(sub["n_replications"].iloc[0]) if "n_replications" in sub.columns else None,
    }