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
    
    # Unicode math replacements (text/header only)
    if is_text or is_header:
        replacements = {
            '≥': r'$\geq$', '≤': r'$\leq$', '–': '--', '—': '---',
            '×': r'$\times$', '±': r'$\pm$', '<': r'$<$', '>': r'$>$',
            '†': r'$\dagger$', '*': r'$^*$'
        }
        for char, repl in replacements.items():
            s = s.replace(char, repl)
        
        # Bold markdown to LaTeX
        if is_text:
            s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    
    # Standard escapes (skip $ if we added math mode)
    skip_dollar = is_header or is_text
    for c in ['&', '#', '{', '}', '_', '%'] + ([] if skip_dollar else ['$']):
        s = s.replace(c, f'\\{c}')
    
    # Try numeric formatting for plain values
    if not (is_header or is_text):
        try: return f"{float(val):.2f}"
        except: pass
    
    return s

def render_latex_table(df: pd.DataFrame, output_path: Path):
    df_out = df.rename(columns=DEMOGRAPHIC_TABLE_COLUMNS)[DEMOGRAPHIC_TABLE_ORDER]
    n_cols = len(DEMOGRAPHIC_TABLE_ORDER)
    
    # Header
    lines = [
        r'\begin{table*}[htbp]', r'\centering', r'\small',
        r'\begin{tabular}{|' + '|'.join(['l'] + ['c'] * (n_cols - 1)) + '|}',
        r'\hline',
        ' & '.join(escape_latex(col, is_header=True) for col in DEMOGRAPHIC_TABLE_ORDER) + r' \\',
        r'\hline'
    ]
    
    # Data rows
    for _, r in df_out.iterrows():
        ci_lower, ci_upper = (f"{r.get(k, 'n/a'):.2f}" if pd.notna(r.get(k)) else "n/a" 
                              for k in ['95% CI* Lower', '95% CI* Upper'])
        pval = (r'$<$0.001' if r.get("p-value†", 1) < 0.001 
                else f"{r.get('p-value†', 0):.3f}")
        
        cells = [
            escape_latex(r["Scenario"]),
            *[f"{r[k]:.2f}" for k in ['Coverage', 'Density', 'SD (Density)', 'Lift']],
            "n/a" if pd.isna(r["Effect Size (d)"]) else f"{r['Effect Size (d)']:.2f}",
            ci_lower, ci_upper, pval,
            f"{r['Stability']:.2f}",
            str(int(r['n_segment']))
        ]
        lines.append(' & '.join(cells) + r' \\')
    
    # Footer
    lines.extend([
        r'\hline', r'\end{tabular}',
        r'\caption{' + escape_latex(DEMOGRAPHIC_TABLE_CAPTION, is_text=True).replace('\n', ' ').strip() + '}',
        r'\end{table*}'
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"📄 LaTeX saved: {output_path}")