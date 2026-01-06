# centralized column mapping for demographic tables

DEMOGRAPHIC_TABLE_COLUMNS = {
    "scenario": "Scenario",
    "coverage": "Coverage",
    "density": "Density",
    "density_sd": "SD (Density)",
    "lift": "Lift",
    "effect_size_d": "Effect Size (d)",
    "effect_ci_lower": "95% CI* Lower",
    "effect_ci_upper": "95% CI* Upper",
    "p_value": "p-value†",
    "stability": "Stability",
    "n_segment": "n_segment"
}

DEMOGRAPHIC_TABLE_ORDER = list(DEMOGRAPHIC_TABLE_COLUMNS.values())
