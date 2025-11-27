import pandas as pd
from pathlib import Path

def generate_demographic_table(csv_path: str,
                               summary_csv_path: str,
                               raw_csv_path: str,
                               output_path: str = "/tmp/demographic_profiles.md"):
    """
    Generate an enriched Markdown table with 95% CI and p-values.
    """

    # Load data
    df = pd.read_csv(csv_path)
    summary_df = pd.read_csv(summary_csv_path)
    raw_df = pd.read_csv(raw_csv_path)

    # 1️⃣ Add 95% CI (same as before)
    ci_df = summary_df.groupby("scenario").agg({
        "density_ci_lower": "mean",
        "density_ci_upper": "mean"
    }).reset_index()
    df = df.merge(ci_df, on="scenario", how="left")
    df["95% CI"] = df.apply(lambda x: f"[{x['density_ci_lower']:.2f}, {x['density_ci_upper']:.2f}]", axis=1)

    # 2️⃣ Compute p-value
    # Use NI baseline density as null
    baseline_density = df.loc[df["scenario"] == "NI", "density"].values[0]

    def compute_pval(scenario):
        densities = raw_df.loc[raw_df["scenario"] == scenario, "density"].values
        # Fraction of replications <= baseline_density
        p_val = (densities <= baseline_density).mean()
        # If all densities are above baseline, set very small value
        return max(p_val, 1e-4)

    df["p-value"] = df["scenario"].apply(lambda s: "<0.001" if compute_pval(s) < 0.001 else f"{compute_pval(s):.3f}")

    # Select columns for table
    columns = [
        "scenario", "segment_name", "coverage", "density", "lift", "95% CI", "p-value",
        "n_agents_total", "n_agents_segment"
    ]
    table_df = df[columns]

    # Convert to Markdown table
    markdown_table = table_df.to_markdown(index=False)

    # Add caption
    caption = (
        "**Patient Rule Induction Method (PRIM) Subgroup Analysis:** Demographic Profiles of High-Adoption Segments Across Policy Scenarios. "
        "Analysis based on 10,000 agents per scenario pooled from 100 independent simulation runs. Coverage indicates proportion of population within each subgroup; "
        "Density represents adoption rate within subgroup; Lift shows ratio of subgroup density to scenario baseline.**\n"
    )

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        f.write(caption + "\n" + markdown_table)

    print(f"Markdown table saved to: {output_path}")


if __name__ == "__main__":
    csv_file = "data/dummy/demographic_profiles.csv"
    summary_csv_file = "data/dummy/prim_trajectory_summary.csv"
    raw_csv_file = "data/dummy/prim_trajectory_raw.csv"
    generate_demographic_table(csv_file, summary_csv_file, raw_csv_file)
