import pandas as pd
from pathlib import Path
from ._utils.file_utils import load_csv_or_fail
from src.utils.cli_parser import base_parser, safe_run

# ----------------------------------------------------
# Statistics Computation
# ----------------------------------------------------

def compute_stability(raw_df, scenario):
    return raw_df.loc[raw_df["scenario"] == scenario, "is_selected"].mean()

def compute_pvalue(raw_df, baseline_density, scenario):
    densities = raw_df.loc[raw_df["scenario"] == scenario, "density"].values
    return max((densities <= baseline_density).mean(), 1e-4)

def compute_cohens_d(raw_df, scenario, baseline="NI"):
    base = raw_df.loc[raw_df["scenario"] == baseline, "density"].values
    comp = raw_df.loc[raw_df["scenario"] == scenario, "density"].values
    if len(base) == 0 or len(comp) == 0:
        return None
    pooled_sd = ((base.std()**2 + comp.std()**2)/2)**0.5
    return None if pooled_sd == 0 else (comp.mean() - base.mean())/pooled_sd

def interpret_effect_size(d):
    if d is None:
        return "n/a"
    d_abs = abs(d)
    return ("negligible" if d_abs < 0.2 else "small" if d_abs < 0.5
            else "medium" if d_abs < 0.8 else "large")

# ----------------------------------------------------
# Data Preparation
# ----------------------------------------------------

def prepare_table(csv_path, summary_csv_path, raw_csv_path):
    df = load_csv_or_fail(csv_path)
    summary_df = load_csv_or_fail(summary_csv_path)
    raw_df = load_csv_or_fail(raw_csv_path)

    # Compute 95% CI
    ci_df = summary_df.groupby("scenario").agg({
        "density_ci_lower": "mean", "density_ci_upper": "mean"
    }).reset_index()
    df = df.merge(ci_df, on="scenario", how="left")
    df["95% CI"] = df.apply(lambda x: f"[{x['density_ci_lower']:.2f}, {x['density_ci_upper']:.2f}]", axis=1)

    baseline_density = df.loc[df["scenario"] == "NI", "density"].values[0]

    # Compute statistics
    df["Stability†"] = df["scenario"].apply(lambda s: f"{compute_stability(raw_df, s):.2f}")
    df["p-value"] = df["scenario"].apply(
        lambda s: "<0.001" if compute_pvalue(raw_df, baseline_density, s) < 0.001
        else f"{compute_pvalue(raw_df, baseline_density, s):.3f}"
    )
    df["Effect Size (d)"] = df["scenario"].apply(lambda s: compute_cohens_d(raw_df, s))
    df["Effect"] = df["Effect Size (d)"].apply(interpret_effect_size)
    df["Effect Size (d)"] = df["Effect Size (d)"].apply(lambda d: f"{d:.2f}" if d else "n/a")

    # Remove segment_name from output
    columns_final = ["scenario", "coverage", "density", "lift",
                     "95% CI", "p-value", "Stability†", "Effect Size (d)", "Effect",
                     "n_agents_total", "n_agents_segment"]
    return df[columns_final]

# ----------------------------------------------------
# Output Writers
# ----------------------------------------------------

def escape_latex(val):
    """Escape LaTeX special chars and format numbers."""
    if pd.isna(val):
        return ""
    s = str(val).replace('≥', r'$\ge$').replace('≤', r'$\le$').replace('†', r'$\dagger$')
    # Escape % e altri caratteri speciali
    for c in ['&', '#', '{', '}', '_']:
        s = s.replace(c, f'\\{c}')
    s = s.replace('%', r'\%')  # speciale per percentuali
    try:
        return f"{float(s):.2f}"
    except:
        return s

def write_latex(df, output_path):
    """
    Writes LaTeX table using tabularx with numeric columns centered.
    segment_name removed; explained in caption.
    Escapes % and usa -- nei range nella caption.
    """
    from textwrap import dedent

    col_headers = [
        "scenario", "coverage", "density", "lift", r"95\% CI", "p-value",
        "Stability$\\dagger$", "Effect Size (d)", "Effect",
        "n agents total", "n agents segment"
    ]

    if len(col_headers) != len(df.columns):
        raise ValueError(f"Column count mismatch: {len(col_headers)} headers vs {len(df.columns)} dataframe columns")

    cols_fmt = '|l|' + '|'.join(['>{\\centering\\arraybackslash}p{1.2cm}']*(len(df.columns)-1)) + '|'

    lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        r'\small',
        r'\begin{tabularx}{\textwidth}{' + cols_fmt + r'}',
        r'\hline',
        ' & '.join(col_headers) + r' \\',
        r'\hline'
    ]

    for _, row in df.iterrows():
        cells = [escape_latex(v) for v in row]
        if len(cells) != len(col_headers):
            raise ValueError(f"Row column count mismatch: {len(cells)} vs {len(col_headers)}")
        lines.append(' & '.join(cells) + r' \\')

    lines += [
        r'\hline',
        r'\end{tabularx}',
        dedent(rf'''
        \caption{{PRIM Subgroup Analysis: demographic profiles across scenarios.
        Segment codes: NI = Baseline Population (No Segmentation), SI = High Trust Community (Trust $\ge$ 0.65), EI = High Trust + Mid-High Income (Trust $\ge$ 0.55, Income $\ge$ 30).
        Stability reflects the average proportion of replications where the subgroup is selected.
        Effect sizes (Cohen's d) quantify the magnitude of difference in adoption density relative to NI baseline.
        Thresholds: negligible (<0.2), small (0.2--0.5), medium (0.5--0.8), large (>0.8).}}
        ''').strip(),
        r'\label{tab:demographic_profiles}',
        r'\end{table*}'
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines))
    print(f"📄 LaTeX saved: {output_path}")
    
def write_markdown(df, output_path):
    caption = (
        "**Patient Rule Induction Method (PRIM) Subgroup Analysis:**\n"
        "Demographic profiles of high-adoption segments across policy scenarios.\n"
        "Segment codes: NI = Baseline Population (No Segmentation), SI = High Trust Community (Trust ≥ 0.65), EI = High Trust + Mid-High Income (Trust ≥ 0.55, Income ≥ 30).\n"
        "Stability reflects the average proportion of replications where the subgroup is selected.\n"
        "Effect sizes (Cohen's d) quantify the magnitude of difference in adoption density relative to the NI baseline.\n"
        "Thresholds: negligible (<0.2), small (0.2–0.5), medium (0.5–0.8), large (>0.8).\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(caption + "\n" + df.to_markdown(index=False))
    print(f"📄 Markdown saved: {output_path}")

# ----------------------------------------------------
# Main
# ----------------------------------------------------

def generate_demographic_table(csv_path, summary_csv_path, raw_csv_path, output_path):
    df = prepare_table(csv_path, summary_csv_path, raw_csv_path)
    write_markdown(df, output_path.with_suffix('.md'))
    write_latex(df, output_path.with_suffix('.tex'))

if __name__ == "__main__":
    safe_run(lambda: (
        (args := base_parser(defaults={"output": Path("/tmp/demographic_profiles.md")}).parse_args()),
        (data_dir := Path(args.data_dir)) or (lambda: (_ for _ in ()).throw(ValueError("❌ --data-dir required")))(),
        generate_demographic_table(
            csv_path=data_dir / "demographic_profiles.csv",
            summary_csv_path=data_dir / "prim_trajectory_summary.csv",
            raw_csv_path=data_dir / "prim_trajectory_raw.csv",
            output_path=Path(args.output)
        )
    ))
