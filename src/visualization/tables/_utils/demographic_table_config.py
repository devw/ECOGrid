# Column mapping and order
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

# Shared caption for both Markdown and LaTeX
DEMOGRAPHIC_TABLE_CAPTION = (
    "**Patient Rule Induction Method (PRIM) Subgroup Analysis:** "
    "Demographic profiles of high-adoption segments identified through algorithmic "
    "box-peeling across 100 independent replications (n=5,000 agents each).\n"
    "**Segment definitions:** NI = Baseline Population (no restrictions); "
    "SI = High Trust Community (Trust ≥ 0.65); "
    "EI = High Trust + Mid-High Income (Trust ≥ 0.55, Income ≥ 30k).\n"
    "**Metrics:** Coverage = proportion of total population in segment; "
    "Density = adoption rate within segment; "
    "Lift = density relative to population mean; "
    "Effect Size = Cohen's d measuring standardized difference from baseline.\n"
    "**Statistical inference:** All estimates derived from 100 replications. "
    "Effect size thresholds follow Cohen (1988): negligible (<0.2), small (0.2–0.5), "
    "medium (0.5–0.8), large (>0.8). "
    "Confidence intervals computed via percentile bootstrap (95% level). "
    "P-values from two-tailed t-tests comparing replication-level densities against baseline.\n"
    "**Stability:** Proportion of PRIM peeling iterations (15 algorithmic steps) "
    "where each subgroup configuration was identified as optimal by the algorithm's "
    "density-coverage objective function."
)

AGGREGATE_TABLE_CAPTION = (
    "**Aggregate Adoption Metrics by Policy Scenario:** "
    "Distributional analysis across 225 trust-income bins (15×15 grid) "
    "from 100 replications (n=5,000 agents each). "
    "Scenarios: NI=Baseline, SI=Social influence intervention, EI=Economic incentives. "
    "Trust threshold (≥0.64) aligns with PRIM segmentation; income stratified by tertiles "
    "(Low <33rd, High ≥67th percentile). "
    "Mean Adoption: population-weighted average; Std Dev: between-bin heterogeneity "
    "(not sampling uncertainty); Income Gap: percentage point difference quantifying inequality. "
    "For replication-level confidence intervals, see detailed tables."
)
