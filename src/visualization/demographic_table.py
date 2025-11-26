import pandas as pd
from pathlib import Path

def generate_demographic_table(csv_path: str, output_path: str = "/tmp/demographic_profiles.md"):
    """
    Generate a Markdown table from demographic_profiles.csv and save it to /tmp/.

    Parameters:
    - csv_path: path to demographic_profiles.csv
    - output_path: where to save the Markdown table
    """

    df = pd.read_csv(csv_path)

    # Select relevant columns for table
    columns = [
        "scenario", "segment_name", "coverage", "density", "lift",
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
    generate_demographic_table(csv_file)
