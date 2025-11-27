import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_prim_trajectory_summary(csv_path: str, output_path: str = "prim_trajectory_updated.png"):
    """
    Generate PRIM Peeling Trajectory (Coverage vs Density) with Error Intervals
    and Key Point Annotations using summary data.

    Parameters:
    - csv_path: path to prim_trajectory_summary.csv file
    - output_path: where to save the plot
    """
    df = pd.read_csv(csv_path)
    
    # Calculate asymmetric density error (95% CI)
    df['density_err_lower'] = df['density_mean'] - df['density_ci_lower']
    df['density_err_upper'] = df['density_ci_upper'] - df['density_mean']
    
    colors = {"NI": "blue", "EI": "green", "SI": "orange"}
    fig, ax = plt.subplots(figsize=(10, 7))

    for scenario in df["scenario"].unique():
        subset = df[df["scenario"] == scenario].sort_values(by="iteration")
        color = colors.get(scenario, "black")
        y_err = np.array([subset['density_err_lower'].values, subset['density_err_upper'].values])

        # Plot trajectory with error bars
        ax.errorbar(
            subset["coverage_mean"], subset["density_mean"], yerr=y_err,
            label=f"{scenario} Trajectory (Mean ± 95% CI)", color=color,
            linewidth=2, marker='o', markersize=4, capsize=3, 
            elinewidth=0.8, zorder=2, errorevery=2
        )

        # Mark selected boxes with stars
        selected = subset[subset["is_selected"]]
        ax.scatter(
            selected["coverage_mean"], selected["density_mean"],
            color=color, marker="*", s=200, edgecolor="black",
            linewidth=1.2, zorder=3
        )
        
        # Annotate key points
        for _, row in selected.iterrows():
            ax.annotate(
                f"Iter {int(row['iteration'])}", 
                (row["coverage_mean"], row["density_mean"]),
                textcoords="offset points", xytext=(5, 15), ha='center',
                fontsize=12, color=color,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", 
                              color=color, linewidth=0.7)
            )

    # Random targeting diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, 
            label="Random Targeting", zorder=1)

    # Labels and styling
    ax.set_xlabel("Coverage (Mean)", fontsize=16)
    ax.set_ylabel("Density (Mean ± 95% CI)", fontsize=16)
    ax.set_title("PRIM Peeling Trajectory: Coverage-Density Tradeoff Across Policy Scenarios", 
                fontsize=18)
    ax.legend(title="Scenario", title_fontsize=14, fontsize=12)
    
    # Set limits with 5% margin
    margin = 0.05
    ax.set_xlim(min(0.0, df["coverage_mean"].min() - margin), 
                max(1.0, df["coverage_mean"].max() + margin))
    ax.set_ylim(min(0.0, df["density_mean"].min() - margin), 
                max(1.0, df["density_mean"].max() + margin))
    ax.grid(alpha=0.3, zorder=0)

    # Save figure
    full_path = Path("/tmp") / output_path
    plt.savefig(full_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    
    print(f"📊 Figure saved at: {full_path}")

if __name__ == "__main__":
    plot_prim_trajectory_summary("data/dummy/prim_trajectory_summary.csv")