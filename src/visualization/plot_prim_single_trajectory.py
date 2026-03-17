import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.cli_parser import base_parser, safe_run
from src.utils.file_utils import load_csv_or_fail
from src.utils.plot_style import (
    SCENARIO_COLORS, SELECTED_BOX_STYLE, ANNOTATION_STYLE
)


def read_csv(csv_path: Path):
    """Read the CSV file and return a DataFrame."""
    return load_csv_or_fail(csv_path)


def create_prim_plot(df, output_path: Path):
    """Generate deterministic PRIM trajectory plot (Metodo A)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    baseline_density = 0.20  # top 20% definition

    for scenario in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scenario].sort_values("iteration")
        color = SCENARIO_COLORS.get(scenario, "black")

        # Plot deterministic trajectory (no error bars)
        ax.plot(
            sub["coverage"],
            sub["density"],
            marker="o",
            label=scenario,
            color=color
        )

        # Highlight selected box
        sel = sub[sub["is_selected"]]
        if not sel.empty:
            row = sel.iloc[0]

            ax.scatter(
                row["coverage"],
                row["density"],
                color=color,
                **SELECTED_BOX_STYLE
            )

            ax.annotate(
                f"Iter {int(row['iteration'])}",
                (row["coverage"], row["density"]),
                xytext=ANNOTATION_STYLE["text_offset"],
                textcoords="offset points",
                fontsize=ANNOTATION_STYLE["fontsize"],
                color=color,
                arrowprops={**ANNOTATION_STYLE["arrowprops"], "color": color}
            )

    # Random targeting baseline (correct PRIM benchmark)
    ax.axhline(
        y=baseline_density,
        linestyle="--",
        color="gray",
        linewidth=1.3,
        label="Random Targeting (20%)"
    )

    ax.set_xlabel("Coverage", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.set_title("PRIM Peeling Trajectory: Coverage–Density Tradeoff", fontsize=17)
    ax.legend(title="Scenario", fontsize=11, title_fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close()

    print(f"✔ Saved plot to: {output_path.resolve()}")


def print_selected_boxes(df):
    """
    Print coverage and density values for the selected PRIM box
    for each scenario (Metodo A).
    """
    print("\n=== SELECTED PRIM BOXES (RUN-LEVEL) ===\n")

    for scenario in sorted(df["scenario"].unique()):
        sub = df[(df["scenario"] == scenario) & (df["is_selected"])]

        if sub.empty:
            print(f"[{scenario}] No selected box found.")
            continue

        row = sub.iloc[0]

        print(f"Scenario: {scenario}")
        print(f"  Iteration: {int(row['iteration'])}")
        print(f"  Coverage:  {row['coverage']:.4f}")
        print(f"  Density:   {row['density']:.4f}")
        print(f"  N runs:    {int(row['n_runs'])}")
        print()


def main():
    args = base_parser(
        defaults={
            "data_dir": Path("data"),
            "output": Path("/tmp/prim_trajectory.png"),
        }
    ).parse_args()

    csv_path = Path(args.data_dir) / "prim_single_trajectory.csv"
    df = read_csv(csv_path)

    print_selected_boxes(df)
    create_prim_plot(df, Path(args.output))


if __name__ == "__main__":
    safe_run(main)