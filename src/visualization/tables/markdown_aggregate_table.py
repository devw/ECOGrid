"""
Render aggregate metrics table in Markdown format.
"""
from pathlib import Path
import pandas as pd
from ._utils.tables_config import AGGREGATE_TABLE_CAPTION


def format_percentage(value: float) -> str:
    """Format value as percentage with 2 decimal places."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}"


def format_float(value: float, decimals: int = 4) -> str:
    """Format float with specified decimals."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def format_int(value: float) -> str:
    """Format as integer."""
    if pd.isna(value):
        return "N/A"
    return f"{int(value)}"


def render_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    """
    Render aggregate metrics table in Markdown format.
    
    Args:
        df: DataFrame with aggregate metrics
        output_path: Path where to save the .md file
    """
    # Prepare formatted data
    formatted_rows = []
    
    for _, row in df.iterrows():
        formatted_rows.append({
            'Scenario': row['scenario'],
            'Mean Adoption (%)': format_percentage(row['mean_adoption_rate']),
            'Std Dev (%)': format_percentage(row['std_adoption_rate']),
            'High-Trust (%)': format_percentage(row['high_trust_adoption']),
            'High-Income (%)': format_percentage(row['high_income_adoption']),
            'Low-Income (%)': format_percentage(row['low_income_adoption']),
            'Income Gap (pp)': format_percentage(row['income_gap']),
            'n_bins': format_int(row['n_bins'])
        })
    
    formatted_df = pd.DataFrame(formatted_rows)
    
    # Generate Markdown
    caption = AGGREGATE_TABLE_CAPTION.replace(":", "\\:")
    table_md = formatted_df.to_markdown(index=False)
    
    # Combine caption and table
    full_markdown = f"{caption}\n\n{table_md}\n"
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(full_markdown)
    
    print(f"📄 Markdown saved: {output_path}")