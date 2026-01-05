from pathlib import Path
import pandas as pd
from textwrap import dedent
from ._utils.demographic_table_config import DEMOGRAPHIC_TABLE_COLUMNS, DEMOGRAPHIC_TABLE_ORDER, DEMOGRAPHIC_TABLE_CAPTION

def escape_latex(val):
    """Escape LaTeX special characters and format numeric values."""
    if pd.isna(val):
        return ""
    s = str(val)
    for c in ['&', '#', '{', '}', '_', '%', '$', '†']:
        s = s.replace(c, f'\\{c}')
    try:
        return f"{float(val):.2f}"
    except Exception:
        return s

def render_latex_table(df: pd.DataFrame, output_path: Path):
    """
    Render a demographic table as LaTeX.

    - Columns and order are taken from DEMOGRAPHIC_TABLE_COLUMNS / ORDER.
    - Effect Size CI is formatted as [lower, upper].
    - p-values <0.001 are shown as "<0.001".
    """
    # Rename and reorder columns
    df_out = df.rename(columns=DEMOGRAPHIC_TABLE_COLUMNS)[DEMOGRAPHIC_TABLE_ORDER]

    col_headers = DEMOGRAPHIC_TABLE_ORDER
    n_cols = len(col_headers)
    
    lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{|' + '|'.join(['l'] + ['c']*(n_cols-1)) + '|}',
        r'\hline',
        ' & '.join(col_headers) + r' \\',
        r'\hline'
    ]

    for _, row in df.iterrows():
        # Cohen's d CI
        ci = (
            f"[{row['effect_ci_lower']:.2f}, {row['effect_ci_upper']:.2f}]"
            if pd.notna(row['effect_ci_lower']) else "n/a"
        )
        # p-value formatting
        pval = "<0.001" if row["p_value"] < 0.001 else f"{row['p_value']:.3f}"
        cells = [
            escape_latex(row["scenario"]),
            f"{row['coverage']:.2f}",
            f"{row['density']:.2f}",
            f"{row['density_sd']:.2f}",
            f"{row['lift']:.2f}",
            "n/a" if pd.isna(row["effect_size_d"]) else f"{row['effect_size_d']:.2f}",
            ci,
            pval,
            f"{row['stability']:.2f}",
            str(int(row["n_segment"]))
        ]
        lines.append(' & '.join(cells) + r' \\')

    lines += [
        r'\hline',
        r'\end{tabular}',
        dedent(f"""
        \\caption{{{DEMOGRAPHIC_TABLE_CAPTION}}}
        """).strip(),
        r'\end{table*}'
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"📄 LaTeX saved: {output_path}")
