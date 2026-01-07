"""
Render aggregate metrics table in LaTeX format.
"""
from pathlib import Path
import pandas as pd


def format_percentage(value: float) -> str:
    """Format value as percentage with 2 decimal places."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}"


def format_int(value: float) -> str:
    """Format as integer."""
    if pd.isna(value):
        return "N/A"
    return f"{int(value)}"


def generate_latex_caption() -> str:
    """Generate the LaTeX caption for the table."""
    return r"""$^*$$^*$Aggregate Adoption Metrics by Policy Scenario:$^*$$^*$ Summary statistics of green energy adoption across demographic segments. Adoption rates are weighted by sample size across trust-income bins. Income brackets: Low ($<$ 33rd percentile), High ($\geq$ 67th percentile). Trust threshold: High trust defined as $\geq$ 0.64 (aligned with PRIM segmentation). Income Gap quantifies adoption inequality between high and low-income populations. Based on 100 Monte Carlo replications with 5,000 agents each."""


def render_latex_table(df: pd.DataFrame, output_path: Path) -> None:
    """
    Render aggregate metrics table in LaTeX format.
    
    Args:
        df: DataFrame with aggregate metrics
        output_path: Path where to save the .tex file
    """
    # Prepare formatted data
    latex_rows = []
    
    for _, row in df.iterrows():
        scenario = row['scenario']
        mean_adop = format_percentage(row['mean_adoption_rate'])
        std_adop = format_percentage(row['std_adoption_rate'])
        high_trust = format_percentage(row['high_trust_adoption'])
        high_income = format_percentage(row['high_income_adoption'])
        low_income = format_percentage(row['low_income_adoption'])
        income_gap = format_percentage(row['income_gap'])
        n_bins = format_int(row['n_bins'])
        
        latex_rows.append(
            f"{scenario} & {mean_adop} & {std_adop} & {high_trust} & {high_income} & {low_income} & {income_gap} & {n_bins} \\\\"
        )
    
    # Generate LaTeX table
    caption = generate_latex_caption()
    
    latex_content = f"""% Requires: \\usepackage{{makecell}}
\\begin{{table*}}[htbp]
\\caption{{{caption}}}
\\label{{tab:aggregate_metrics_by_scenario}}
\\centering
\\small
\\begin{{tabular}}{{|l|c|c|c|c|c|c|c|}}
\\hline
\\makecell{{Scenario}} & \\makecell{{Mean \\\\ Adoption \\\\ (\\%)}} & \\makecell{{Std Dev \\\\ (\\%)}} & \\makecell{{High-Trust \\\\ (\\%)}} & \\makecell{{High-Income \\\\ (\\%)}} & \\makecell{{Low-Income \\\\ (\\%)}} & \\makecell{{Income Gap \\\\ (pp)}} & \\makecell{{n \\\\ bins}} \\\\
\\hline
"""
    
    # Add data rows
    for row in latex_rows:
        latex_content += row + "\n"
    
    # Close table
    latex_content += """\\hline
\\end{tabular}
\\end{table*}
"""
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    print(f"📄 LaTeX saved: {output_path}")
    print("⚠️  Add \\usepackage{makecell} to your preamble")