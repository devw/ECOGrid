from pathlib import Path
import pandas as pd
import re
from ._utils.demographic_table_config import (
    DEMOGRAPHIC_TABLE_COLUMNS, DEMOGRAPHIC_TABLE_ORDER, DEMOGRAPHIC_TABLE_CAPTION
)

def escape_latex(val, is_header=False, is_text=False):
    """Universal LaTeX escaper with context-aware replacements"""
    if pd.isna(val): return ""
    s = str(val)
    
    if is_text or is_header:
        replacements = {
            '≥': r'$\geq$', '≤': r'$\leq$', '–': '--', '—': '---',
            '×': r'$\times$', '±': r'$\pm$', '<': r'$<$', '>': r'$>$',
            '†': r'$\dagger$', '*': r'$^*$'
        }
        for char, repl in replacements.items():
            s = s.replace(char, repl)
        if is_text:
            s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    
    skip_dollar = is_header or is_text
    for c in ['&', '#', '{', '}', '_', '%'] + ([] if skip_dollar else ['$']):
        s = s.replace(c, f'\\{c}')
    
    if not (is_header or is_text):
        try: return f"{float(val):.2f}"
        except: pass
    return s

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
    df_out = df.rename(columns=DEMOGRAPHIC_TABLE_COLUMNS)[DEMOGRAPHIC_TABLE_ORDER]
    n_cols = len(DEMOGRAPHIC_TABLE_ORDER)
    
    lines = [
        r'% Requires: \usepackage{makecell}',
        r'\begin{table*}[htbp]',
        r'\caption{' + escape_latex(DEMOGRAPHIC_TABLE_CAPTION, is_text=True).replace('\n', ' ').strip() + '}',
        r'\label{tab:demographic_profiles}',
        r'\centering', r'\small',
        r'\begin{tabular}{|' + '|'.join(['l'] + ['c'] * (n_cols - 1)) + '|}',
        r'\hline'
    ]
    
    # Headers with multiline wrapping
    headers = [escape_latex(DEMOGRAPHIC_TABLE_ORDER[0], is_header=True)]
    headers.extend(wrap_header(col) for col in DEMOGRAPHIC_TABLE_ORDER[1:])
    lines.append(' & '.join(headers) + r' \\')
    lines.append(r'\hline')
    
    # Data rows
    for _, r in df_out.iterrows():
        ci_low = f"{r.get('95% CI* Lower', 0):.2f}" if pd.notna(r.get('95% CI* Lower')) else "n/a"
        ci_up = f"{r.get('95% CI* Upper', 0):.2f}" if pd.notna(r.get('95% CI* Upper')) else "n/a"
        pval = r.get("p-value†", "1.000")
        if not isinstance(pval, str):
            pval = r'$<$0.001' if pval < 0.001 else f"{pval:.3f}"
        
        cells = [
            escape_latex(r["Scenario"]),
            *[f"{r[k]:.2f}" for k in ['Coverage', 'Density', 'SD (Density)', 'Lift']],
            "n/a" if pd.isna(r.get("Effect Size (d)")) else f"{r['Effect Size (d)']:.2f}",
            ci_low, ci_up, pval,
            f"{r['Stability']:.2f}",
            str(int(r['n_segment']))
        ]
        lines.append(' & '.join(cells) + r' \\')
    
    lines.extend([
        r'\hline', r'\end{tabular}',
        r'\end{table*}'
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"📄 LaTeX saved: {output_path}")
    print(f"⚠️  Add \\usepackage{{makecell}} to your preamble")