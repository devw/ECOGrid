import pandas as pd
from pathlib import Path


def generate_demographic_table(
    csv_path: str,
    summary_csv_path: str,
    raw_csv_path: str,
    output_path: str = "/tmp/demographic_profiles.md"
):
    """
    Generate an enriched Markdown table with 95% CI, p-values, and Stability†.
    """

    # Load data
    df = pd.read_csv(csv_path)
    summary_df = pd.read_csv(summary_csv_path)
    raw_df = pd.read_csv(raw_csv_path)

    # --- Helper functions MUST be defined before use ---

    def compute_stability(scenario: str) -> float:
        replications = raw_df.loc[raw_df["scenario"] == scenario, "is_selected"]
        return replications.mean()

    def compute_pval(scenario: str) -> float:
        densities = raw_df.loc[raw_df["scenario"] == scenario, "density"].values
        p_val = (densities <= baseline_density).mean()
        return max(p_val, 1e-4)

    # 1️⃣ Add 95% CI
    ci_df = (
        summary_df.groupby("scenario")
        .agg({
            "density_ci_lower": "mean",
            "density_ci_upper": "mean"
        })
        .reset_index()
    )

    df = df.merge(ci_df, on="scenario", how="left")

    df["95% CI"] = df.apply(
        lambda x: f"[{x['density_ci_lower']:.2f}, {x['density_ci_upper']:.2f}]",
        axis=1
    )

    # 2️⃣ Stability†
    df["Stability†"] = df["scenario"].apply(
        lambda s: f"{compute_stability(s):.2f}"
    )

    # 3️⃣ p-value
    baseline_density = df.loc[df["scenario"] == "NI", "density"].iloc[0]

    df["p-value"] = df["scenario"].apply(
        lambda s: "<0.001" if compute_pval(s) < 0.001 else f"{compute_pval(s):.3f}"
    )

    # Select final columns
    columns = [
        "scenario", "segment_name", "coverage", "density", "lift",
        "95% CI", "p-value", "Stability†",
        "n_agents_total", "n_agents_segment"
    ]

    table_df = df[columns]

    # Markdown output
    markdown_table = table_df.to_markdown(index=False)

    caption = (
        "**Patient Rule Induction Method (PRIM) Subgroup Analysis:** "
        "Demographic Profiles of High-Adoption Segments Across Policy Scenarios. "
        "Analysis based on 10,000 agents per scenario pooled from 100 independent simulation runs. "
        "Coverage indicates proportion of population within each subgroup; "
        "Density represents adoption rate within subgroup; "
        "Lift shows ratio of subgroup density to scenario baseline.**\n"
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
