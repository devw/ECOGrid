from pathlib import Path
import pandas as pd

def render_markdown_table(df: pd.DataFrame, output_path: Path):
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

    caption = (
        "**Patient Rule Induction Method (PRIM) Subgroup Analysis:**\n"
        "Demographic profiles of high-adoption segments across policy scenarios.\n"
        "Segment codes: NI = Baseline Population (No Segmentation), SI = High Trust Community (Trust ≥ 0.65), EI = High Trust + Mid-High Income (Trust ≥ 0.55, Income ≥ 30).\n"
        "Stability reflects the average proportion of replications where the subgroup is selected.\n"
        "Effect sizes (Cohen's d) quantify the magnitude of difference in adoption density relative to the NI baseline.\n"
        "Thresholds: negligible (<0.2), small (0.2–0.5), medium (0.5–0.8), large (>0.8).\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(caption + "\n" + df_out.to_markdown(index=False))
    print(f"📄 Markdown saved: {output_path}")
