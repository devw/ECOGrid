import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.cli_parser import base_parser, safe_run
from src.utils.file_utils import load_csv_or_fail
from src.utils.plot_style import (
    SCENARIO_COLORS, ERRORBAR_STYLE, SELECTED_BOX_STYLE, ANNOTATION_STYLE
)

def plot_prim_trajectory_summary(csv_path, output_path="/tmp/prim_trajectory.png"):
    df = load_csv_or_fail(csv_path)
    fig, ax = plt.subplots(figsize=(10, 7))

    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario].sort_values("iteration")
        color = SCENARIO_COLORS.get(scenario, "black")

        yerr = np.vstack([
            sub["density_mean"] - sub["density_ci_lower"],
            sub["density_ci_upper"] - sub["density_mean"]
        ])

        ax.errorbar(
            sub["coverage_mean"], sub["density_mean"],
            yerr=yerr, color=color, label=scenario, **ERRORBAR_STYLE
        )

        sel = sub[sub["is_selected"]]
        if len(sel):
            ax.scatter(sel["coverage_mean"], sel["density_mean"], color=color, **SELECTED_BOX_STYLE)
            for _, r in sel.iterrows():
                ax.annotate(
                    f"Iter {int(r['iteration'])}",
                    (r["coverage_mean"], r["density_mean"]),
                    xytext=ANNOTATION_STYLE["text_offset"],
                    textcoords="offset points",
                    fontsize=ANNOTATION_STYLE["fontsize"],
                    color=color,
                    arrowprops={**ANNOTATION_STYLE["arrowprops"], "color": color}
                )

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.3, label="Random Targeting")
    ax.set_xlabel("Coverage (Mean)", fontsize=15)
    ax.set_ylabel("Density (Mean ± 95% CI)", fontsize=15)
    ax.set_title("PRIM Peeling Trajectory: Coverage–Density Tradeoff", fontsize=17)
    ax.legend(title="Scenario", fontsize=11, title_fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close()
    print(f"✔ Saved: {output_path.resolve()}")

def main():
    args = base_parser(
        defaults={
            "data_dir": Path("data/montecarlo"),
            "output": Path("/tmp/prim_trajectory.png"),
        }
    ).parse_args()

    csv = Path(args.data_dir) / "prim_trajectory_summary.csv"
    plot_prim_trajectory_summary(csv, args.output)

if __name__ == "__main__":
    safe_run(main)
