from pathlib import Path
import pandas as pd
from ._utils.tables_config import (DEMOGRAPHIC_TABLE_COLUMNS, DEMOGRAPHIC_TABLE_ORDER, DEMOGRAPHIC_TABLE_CAPTION)
from ._utils.latex import escape_latex

def wrap_header(text: str) -> str:
    """Wrap long headers with \\makecell for multiline display"""
    # Direct string replacements (no regex to avoid escape issues)
    breaks = {
        'Effect Size (d)': 'Effect Size \\\\ (d)',
        '95% CI* Lower': '95\\% CI$^*$ \\\\ Lower',
        '95% CI* Upper': '95\\% CI$^*$ \\\\ Upper',
        'SD (Density)': 'SD \\\\ (Density)',
        'p-value†': 'p-value \\\\ $\\dagger$',
        'n_segment': 'n \\\\ segment'
    }
    result = breaks.get(text, escape_latex(text, is_header=True))
    return rf'\makecell{{{result}}}'

def render_latex_table(df: pd.DataFrame, output_path: Path):

    latex = df.to_latex(
        index=False,
        float_format="%.3f",
        escape=False
    )

    with open(output_path, "w") as f:
        f.write(latex)
