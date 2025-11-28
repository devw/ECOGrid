import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ._utils.file_utils import load_csv_or_fail

def plot_prim_trajectory_summary(
    csv_path: Path | str, 
    output_path: Path | str = "/tmp/prim_trajectory.png"
):
    """
    Generate PRIM Peeling Trajectory (Coverage vs Density) with error intervals
    and key point annotations using the provided CSV summary data.

    Parameters:
    - csv_path: path to prim_trajectory_summary.csv file
    - output_path: path to save the plot
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    df = load_csv_or_fail(csv_path)

    # Calculate asymmetric density error (95% CI)
    df['density_err_lower'] = df['density_mean'] - df['density_ci_lower']
    df['density_err_upper'] = df['density_ci_upper'] - df['density_mean']

    colors = {"NI": "blue", "EI": "green", "SI": "orange"}
    fig, ax = plt.subplots(figsize=(10, 7))

    for scenario in df["scenario"].unique():
        subset = df[df["scenario"] == scenario].sort_values(by="iteration")
        color = colors.get(scenario, "black")
        y_err = np.array([subset['density_err_lower'].values, subset['density_err_upper'].values])

        ax.errorbar(
            subset["coverage_mean"], subset["density_mean"], yerr=y_err,
            label=f"{scenario} Trajectory (Mean ± 95% CI)", color=color,
            linewidth=2, marker='o', markersize=4, capsize=6, 
            elinewidth=1.5, zorder=2
        )

        # Highlight selected boxes
        selected = subset[subset["is_selected"]]
        ax.scatter(
            selected["coverage_mean"], selected["density_mean"],
            color=color, marker="*", s=500, edgecolor="gray",
            linewidth=0.1, zorder=3
        )

        # Annotate key points
        for _, row in selected.iterrows():
            ax.annotate(
                f"Iteration {int(row['iteration'])}",
                (row["coverage_mean"], row["density_mean"]),
                textcoords="offset points", xytext=(5, 15), ha='center',
                fontsize=12, color=color,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", 
                                color=color, linewidth=0.7)
            )

    # Random targeting diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Random Targeting", zorder=1)

    # Labels, title, legend
    ax.set_xlabel("Coverage (Mean)", fontsize=16)
    ax.set_ylabel("Density (Mean ± 95% CI)", fontsize=16)
    ax.set_title("PRIM Peeling Trajectory: Coverage-Density Tradeoff Across Policy Scenarios", fontsize=18)
    ax.legend(title="Scenario", title_fontsize=14, fontsize=12)

    # Axis limits with 5% margin
    margin = 0.05
    ax.set_xlim(max(0.0, df["coverage_mean"].min() - margin), min(1.0, df["coverage_mean"].max() + margin))
    ax.set_ylim(max(0.0, df["density_mean"].min() - margin), min(1.0, df["density_mean"].max() + margin))
    ax.grid(alpha=0.3, zorder=0)

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Figure saved at: {output_path.resolve()}")


if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Plot PRIM Peeling Trajectory")
        parser.add_argument("--data-dir", type=str, default="data/montecarlo", help="Path to data directory")
        parser.add_argument("--output", type=str, default="/tmp/prim_trajectory.png", help="Output figure path")
        args = parser.parse_args()

        csv_file = Path(args.data_dir) / "prim_trajectory_summary.csv"
        plot_prim_trajectory_summary(csv_file, args.output)
    except FileNotFoundError as e:
        import sys
        print(f"❌ {e}")
        sys.exit(1)  # Exit immediately without traceback
