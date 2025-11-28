import pandas as pd
from pathlib import Path


# ----------------------------------------------------
# 1. Utility functions
# ----------------------------------------------------

def compute_stability(raw_df, scenario):
    replications = raw_df.loc[raw_df["scenario"] == scenario, "is_selected"]
    return replications.mean()


def compute_pvalue(raw_df, baseline_density, scenario):
    densities = raw_df.loc[raw_df["scenario"] == scenario, "density"].values
    p_val = (densities <= baseline_density).mean()
    return max(p_val, 1e-4)


def compute_cohens_d(df, raw_df, scenario, baseline_scenario="NI"):
    """Cohen's d between scenario density distribution and baseline NI."""

    # Extract densities
    base = raw_df.loc[raw_df["scenario"] == baseline_scenario, "density"].values
    comp = raw_df.loc[raw_df["scenario"] == scenario, "density"].values

    if len(base) == 0 or len(comp) == 0:
        return None

    # Means
    mean_base = base.mean()
    mean_comp = comp.mean()

    # Pooled standard deviation
    sd_base = base.std()
    sd_comp = comp.std()
    pooled_sd = ((sd_base**2 + sd_comp**2) / 2)**0.5

    if pooled_sd == 0:
        return None

    return (mean_comp - mean_base) / pooled_sd


def interpret_effect_size(d):
    """Cohen's conventional interpretation."""
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


# ----------------------------------------------------
# 2. Main table generator
# ----------------------------------------------------

def generate_demographic_table(csv_path: str,
                               summary_csv_path: str,
                               raw_csv_path: str,
                               output_path: str = "/tmp/demographic_profiles.md"):
    """Generate an enriched Markdown table with CI, p-values, stability, and effect size."""

    # Load
    df = pd.read_csv(csv_path)
    summary_df = pd.read_csv(summary_csv_path)
    raw_df = pd.read_csv(raw_csv_path)

    # Compute CI
    ci_df = summary_df.groupby("scenario").agg({
        "density_ci_lower": "mean",
        "density_ci_upper": "mean"
    }).reset_index()

    df = df.merge(ci_df, on="scenario", how="left")
    df["95% CI"] = df.apply(lambda x: f"[{x['density_ci_lower']:.2f}, {x['density_ci_upper']:.2f}]", axis=1)

    # Compute baseline
    baseline_density = df.loc[df["scenario"] == "NI", "density"].values[0]

    # Add Stability†
    df["Stability†"] = df["scenario"].apply(lambda s: f"{compute_stability(raw_df, s):.2f}")

    # Add p-value
    df["p-value"] = df["scenario"].apply(
        lambda s: "<0.001" if compute_pvalue(raw_df, baseline_density, s) < 0.001
        else f"{compute_pvalue(raw_df, baseline_density, s):.3f}"
    )

    # Add Cohen's d
    df["Effect Size (d)"] = df["scenario"].apply(lambda s: compute_cohens_d(df, raw_df, s))
    df["Effect"] = df["Effect Size (d)"].apply(
        lambda d: interpret_effect_size(d)
    )

    # Round Cohen’s d for display
    df["Effect Size (d)"] = df["Effect Size (d)"].apply(lambda d: f"{d:.2f}" if d is not None else "n/a")

    # Select columns
    columns = [
        "scenario", "segment_name", "coverage", "density", "lift",
        "95% CI", "p-value", "Stability†",
        "Effect Size (d)", "Effect",
        "n_agents_total", "n_agents_segment"
    ]

    table_df = df[columns]

    # Convert to Markdown
    markdown_table = table_df.to_markdown(index=False)

    # Caption auto-updated with effect size info
    caption = (
        "**Patient Rule Induction Method (PRIM) Subgroup Analysis:**\n"
        "Demographic profiles of high-adoption segments across policy scenarios.\n"
        "Stability reflects the average proportion of replications where the subgroup is selected.\n"
        "Effect sizes (Cohen’s d) quantify the magnitude of difference in adoption density relative to the NI baseline "
        "and follow the thresholds: negligible (<0.2), small (0.2–0.5), medium (0.5–0.8), large (>0.8).\n"
    )

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        f.write(caption + "\n" + markdown_table)

    print(f"Markdown table saved to: {output_path}")


# ----------------------------------------------------
# 3. Script entry point
# ----------------------------------------------------

if __name__ == "__main__":
    csv_file = "data/montecarlo/demographic_profiles.csv"
    summary_csv_file = "data/montecarlo/prim_trajectory_summary.csv"
    raw_csv_file = "data/montecarlo/prim_trajectory_raw.csv"
    generate_demographic_table(csv_file, summary_csv_file, raw_csv_file)
