from pathlib import Path
import pandas as pd
from textwrap import dedent

def escape_latex(val):
    """Escape caratteri speciali LaTeX e formatta numeri"""
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
    Scrive LaTeX table per demographic table.
    Formatta numeri e usa colonne già numeriche.
    """
    column_mapping = {
        "scenario": "Scenario",
        "coverage": "Coverage",
        "density": "Density",
        "density_sd": "SD (Density)",
        "lift": "Lift",
        "effect_size_d": "Effect Size (d)",
        "effect_ci_lower": "95% CI*",
        "effect_ci_upper": "95% CI*",
        "p_value": "p-value†",
        "stability": "Stability",
        "n_segment": "n_segment"
    }

    df_out = df.rename(columns=column_mapping)
    df_out = df_out[list(column_mapping.values())]

    col_headers = list(column_mapping.values())

    # costruzione righe
    lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{|' + '|'.join(['l'] + ['c']*(len(col_headers)-1)) + '|}',
        r'\hline',
        ' & '.join(col_headers) + r' \\',
        r'\hline'
    ]

    for _, row in df.iterrows():
        ci = f"[{row['effect_ci_lower']:.2f}, {row['effect_ci_upper']:.2f}]" if pd.notna(row['effect_ci_lower']) else "n/a"
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
        dedent(r"""
        \caption{Demographic table. Effect Size (Cohen's d) vs baseline NI. *95% CI from bootstrap. †p-value vs baseline.}
        """).strip(),
        r'\end{table*}'
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"📄 LaTeX saved: {output_path}")
