import re
import pandas as pd

def escape_latex(val, is_header=False, is_text=False):
    """Universal LaTeX escaper with context-aware replacements."""
    if pd.isna(val):
        return ""

    s = str(val)

    if is_text or is_header:
        replacements = {
            '≥': r'$\geq$',
            '≤': r'$\leq$',
            '–': '--',
            '—': '---',
            '×': r'$\times$',
            '±': r'$\pm$',
            '<': r'$<$',
            '>': r'$>$',
            '†': r'$\dagger$',
        }
        for char, repl in replacements.items():
            s = s.replace(char, repl)

        if is_text:
            # Markdown bold → LaTeX bold
            s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
            # Remove any leftover '**'
            s = s.replace('**', '')

    # IMPORTANT FIX: do NOT escape braces in text/header
    escape_chars = ['&', '#', '_', '%']
    if not (is_text or is_header):
        escape_chars += ['{', '}', '$']

    for c in escape_chars:
        s = s.replace(c, f'\\{c}')

    if not (is_header or is_text):
        try:
            return f"{float(val):.2f}"
        except Exception:
            pass

    return s
