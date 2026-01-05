import pandas as pd
from pathlib import Path
from ._utils.file_utils import load_csv_or_fail
from src.utils.cli_parser import base_parser, safe_run

# ----------------------------------------------------
# 1. Utility functions
# ----------------------------------------------------

def compute_stability(raw_df, scenario):
    replications = raw_df.loc[raw_df["scenario"] == scenario, "is_selected"]
    return replications.mean()


def compute_pvalue(raw_df, baseline_density, scenario):
    densities = raw_df.loc[raw_df["scenario"] == scenario, "density"].values
    p_val = (densities <= baseline_density).mean()
    return max(p_val, 1e-4)


def compute_cohens_d(df, raw_df, scenario, baseline_scenario="NI"):
    base = raw_df.loc[raw_df["scenario"] == baseline_scenario, "density"].values
    comp = raw_df.loc[raw_df["scenario"] == scenario, "density"].values
    if len(base) == 0 or len(comp) == 0:
        return None
    pooled_sd = ((base.std()**2 + comp.std()**2) / 2)**0.5
    if pooled_sd == 0:
        return None
    return (comp.mean() - base.mean()) / pooled_sd


def interpret_effect_size(d):
    if d is None:
        return "n/a"
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# ----------------------------------------------------
# 2. Main table generator
# ----------------------------------------------------

def generate_demographic_table(csv_path: Path,
                               summary_csv_path: Path,
                               raw_csv_path: Path,
                               output_path: Path):
    """Generate an enriched Markdown table with CI, p-values, stability, and effect size."""

    # Load CSVs using shared utility
    df = load_csv_or_fail(csv_path)
    summary_df = load_csv_or_fail(summary_csv_path)
    raw_df = load_csv_or_fail(raw_csv_path)

    # Compute CI
    ci_df = summary_df.groupby("scenario").agg({
        "density_ci_lower": "mean",
        "density_ci_upper": "mean"
    }).reset_index()
    df = df.merge(ci_df, on="scenario", how="left")
    df["95% CI"] = df.apply(lambda x: f"[{x['density_ci_lower']:.2f}, {x['density_ci_upper']:.2f}]", axis=1)

    # Compute baseline density
    baseline_density = df.loc[df["scenario"] == "NI", "density"].values[0]

    # Add Stability†, p-value, Cohen's d
    df["Stability†"] = df["scenario"].apply(lambda s: f"{compute_stability(raw_df, s):.2f}")
    df["p-value"] = df["scenario"].apply(
        lambda s: "<0.001" if compute_pvalue(raw_df, baseline_density, s) < 0.001
        else f"{compute_pvalue(raw_df, baseline_density, s):.3f}"
    )
    df["Effect Size (d)"] = df["scenario"].apply(lambda s: compute_cohens_d(df, raw_df, s))
    df["Effect"] = df["Effect Size (d)"].apply(lambda d: interpret_effect_size(d))
    df["Effect Size (d)"] = df["Effect Size (d)"].apply(lambda d: f"{d:.2f}" if d is not None else "n/a")

    # Select columns
    columns = [
        "scenario", "segment_name", "coverage", "density", "lift",
        "95% CI", "p-value", "Stability†",
        "Effect Size (d)", "Effect",
        "n_agents_total", "n_agents_segment"
    ]
    table_df = df[columns]

    # Convert to Markdown
    markdown_table = table_df.to_markdown(index=False)

    # Caption
    caption = (
        "**Patient Rule Induction Method (PRIM) Subgroup Analysis:**\n"
        "Demographic profiles of high-adoption segments across policy scenarios.\n"
        "Stability reflects the average proportion of replications where the subgroup is selected.\n"
        "Effect sizes (Cohen’s d) quantify the magnitude of difference in adoption density relative to the NI baseline.\n"
        "Thresholds: negligible (<0.2), small (0.2–0.5), medium (0.5–0.8), large (>0.8).\n"
    )

    # Write to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(caption + "\n" + markdown_table)

    print(f"📄 Markdown table saved to: {output_path}")


if __name__ == "__main__":
    safe_run(lambda: (
        (args := base_parser(defaults={"output": Path("/tmp/demographic_profiles.md")}).parse_args()),
        (data_dir := Path(args.data_dir)) or (lambda: (_ for _ in ()).throw(ValueError("❌ --data-dir argument is required")))(),
        (output_path := Path(args.output)),
        generate_demographic_table(
            csv_path=data_dir / "demographic_profiles.csv",
            summary_csv_path=data_dir / "prim_trajectory_summary.csv",
            raw_csv_path=data_dir / "prim_trajectory_raw.csv",
            output_path=output_path
        )
    ))