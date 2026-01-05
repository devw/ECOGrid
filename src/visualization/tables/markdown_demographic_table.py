from pathlib import Path
import pandas as pd

def render_markdown_table(df: pd.DataFrame, output_path: Path):
    """
    Scrive Markdown table per demographic table.
    Formatta numeri secondo lo stesso schema del LaTeX.
    """
    df_out = df.copy()
    df_out["Effect Size (d)"] = df_out["effect_size_d"].apply(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}")
    df_out["95% CI*"] = df_out.apply(
        lambda r: f"[{r['effect_ci_lower']:.2f}, {r['effect_ci_upper']:.2f}]" if pd.notna(r['effect_ci_lower']) else "n/a",
        axis=1
    )
    df_out["p-value†"] = df_out["p_value"].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
    df_out["Coverage"] = df_out["coverage"].apply(lambda x: f"{x:.2f}")
    df_out["Density"] = df_out["density"].apply(lambda x: f"{x:.2f}")
    df_out["SD (Density)"] = df_out["density_sd"].apply(lambda x: f"{x:.2f}")
    df_out["Lift"] = df_out["lift"].apply(lambda x: f"{x:.2f}")
    df_out["Stability"] = df_out["stability"].apply(lambda x: f"{x:.2f}")
    df_out["n_segment"] = df_out["n_segment"].apply(int)

    columns = ["scenario", "Coverage", "Density", "SD (Density)", "Lift",
               "Effect Size (d)", "95% CI*", "p-value†", "Stability", "n_segment"]

    df_out = df_out.rename(columns={"scenario": "Scenario"})
    df_out = df_out[columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("**Demographic Table:**\n\n" + df_out.to_markdown(index=False))
    print(f"📄 Markdown saved: {output_path}")
