"""
Adoption Heatmap Generator (Refactored)

Generates heatmaps of adoption rate vs. trust & income for multiple scenarios.
Highlights PRIM boxes where relevant.

Output:
    /tmp/adoption_heatmaps_rowlayout.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as Rectangle

DATA_DIR = Path("data/dummy")
HEATMAP_FILE = "heatmap_grid.csv"
PRIM_BOXES_FILE = "prim_boxes.csv"

SCENARIOS = {
    "NI": "No Incentive",
    "SI": "Services Incentive",
    "EI": "Economic Incentive"
}

CMAP = "viridis"
PRIM_COLOR = "yellow"
PRIM_WIDTH = 2


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def scenario_grid(df: pd.DataFrame, s: str):
    data = df[df["scenario"] == s]
    trust = sorted(data["trust_bin"].unique())
    income = sorted(data["income_bin"].unique())

    grid = data.pivot_table(
        index="income_bin",
        columns="trust_bin",
        values="adoption_rate",
        fill_value=0
    ).reindex(index=income, columns=trust).values

    return np.array(trust), np.array(income), grid


def prim_box_patch(box, trust, income):
    """Create a rectangle highlighting a PRIM box."""
    def idx(arr, val): return np.searchsorted(arr, val)

    x0 = idx(trust, box["trust_min"]) - 0.5
    y0 = idx(income, box["income_min"]) - 0.5
    w = idx(trust, box["trust_max"]) - idx(trust, box["trust_min"])
    h = idx(income, box["income_max"]) - idx(income, box["income_min"])

    return Rectangle.Rectangle(
        (x0, y0), w, h,
        fill=False, edgecolor=PRIM_COLOR, linewidth=PRIM_WIDTH, zorder=10
    )


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_heatmap(ax, trust, income, grid, title, prim):
    """Plot a single scenario heatmap."""
    im = ax.imshow(
        grid, origin="lower", aspect="auto", cmap=CMAP,
        vmin=0, vmax=1,
        extent=[-0.5, len(trust)-0.5, -0.5, len(income)-0.5]
    )

    if prim is not None:
        ax.add_patch(prim_box_patch(prim, trust, income))

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Trust")
    ax.set_ylabel("Income")

    # Ticks
    xt = np.linspace(0, len(trust)-1, min(5, len(trust))).astype(int)
    yt = np.linspace(0, len(income)-1, min(5, len(income))).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{trust[i]:.2f}" for i in xt])
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{income[i]:.0f}" for i in yt])

    return im


def plot_all(output: Path):
    heatmap = load_csv(HEATMAP_FILE)
    prim_boxes = load_csv(PRIM_BOXES_FILE)

    # Preload data for scenarios
    data = {
        s: {
            "bins": scenario_grid(heatmap, s),
            "prim": prim_boxes[prim_boxes["scenario"] == s].iloc[0]
                   if not prim_boxes[prim_boxes["scenario"] == s].empty else None
        }
        for s in SCENARIOS
    }

    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(7, 4 * len(SCENARIOS)))
    axes = np.atleast_1d(axes)

    # First scenario → for colorbar
    first = next(iter(SCENARIOS))
    trust, income, grid = data[first]["bins"]
    im = plot_heatmap(axes[0], trust, income, grid, SCENARIOS[first], data[first]["prim"])

    # Rest
    for ax, s in zip(axes[1:], list(SCENARIOS)[1:]):
        trust, income, grid = data[s]["bins"]
        plot_heatmap(ax, trust, income, grid, SCENARIOS[s], data[s]["prim"])

    fig.suptitle("Adoption Rate Across Trust and Income", fontweight="bold", y=0.97)

    # Colorbar above plots
    cax = fig.add_axes([0.15, 0.93, 0.7, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="Adoption Rate")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")

    return fig


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    output = Path("/tmp/adoption_heatmaps_rowlayout.png")
    plot_all(output)
    print(output)


if __name__ == "__main__":
    main()
