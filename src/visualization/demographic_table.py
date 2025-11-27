import pandas as pd
from pathlib import Path

def generate_demographic_table(csv_path: str, summary_csv_path: str,
                               output_path: str = "/tmp/demographic_profiles.md"):
    """
    Generate an enriched Markdown table with 95% CI for density.
    
    Parameters:
    - csv_path: path to demographic_profiles.csv
    - summary_csv_path: path to prim_trajectory_summary.csv
    - output_path: where to save the Markdown table
    """

    # Load data
    df = pd.read_csv(csv_path)
    summary_df = pd.read_csv(summary_csv_path)

    # Compute 95% CI per scenario-segment
    ci_df = summary_df.groupby("scenario").agg({
        "density_ci_lower": "mean",
        "density_ci_upper": "mean"
    }).reset_index()

    # Merge CI into main table
    df = df.merge(ci_df, on="scenario", how="left")

    # Create CI column as string
    df["95% CI"] = df.apply(lambda x: f"[{x['density_ci_lower']:.2f}, {x['density_ci_upper']:.2f}]", axis=1)

    # Select columns for table
    columns = [
        "scenario", "segment_name", "coverage", "density", "lift", "95% CI",
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
    generate_demographic_table(csv_file, summary_csv_file)
