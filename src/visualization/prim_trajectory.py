import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_prim_trajectory(csv_path: str, output_path: str = "figura2_prim_trajectory.png"):
    """
    Generate Figura 2: PRIM Peeling Trajectory (Coverage vs Density).

    Parameters:
    - csv_path: path to prim_trajectory.csv
    - output_path: where to save the plot
    """

    df = pd.read_csv(csv_path)

    # Group by scenario
    scenarios = df["scenario"].unique()

    # Colors assigned to scenarios
    colors = {"NI": "blue", "EI": "green", "SI": "orange"}

    plt.figure(figsize=(10, 7))

    for scenario in scenarios:
        subset = df[df["scenario"] == scenario]
        plt.plot(
            subset["coverage"],
            subset["density"],
            label=scenario,
            color=colors.get(scenario, "black"),
            linewidth=2
        )

        # Mark selected boxes with stars
        selected = subset[subset["is_selected"] == True]
        plt.scatter(
            selected["coverage"],
            selected["density"],
            color=colors.get(scenario, "black"),
            marker="*",
            s=200,
            edgecolor="black",
            linewidth=1.2,
        )

    # Random targeting diagonal line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)

    # Labels and title
    plt.xlabel("Coverage", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title(
        "PRIM Peeling Trajectory: Coverage-Density Trade-off Across Policy Scenarios",
        fontsize=16,
    )
    plt.legend(title="Scenario")
    plt.grid(alpha=0.3)

    # Save figure
    output_path = Path(output_path)
    # Save figure to /tmp/
    output_path = Path("/tmp") / output_path.name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    csv_file = "data/dummy/prim_trajectory.csv"
    plot_prim_trajectory(csv_file)
