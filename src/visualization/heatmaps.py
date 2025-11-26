"""
Heatmap visualization module for adoption rate analysis.

This module generates heatmaps showing adoption rates as a function of trust
and income across different policy scenarios, with PRIM boxes overlaid to
highlight critical parameter subspaces.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Constants
DATA_DIR = Path("data/dummy")
HEATMAP_FILE = "heatmap_grid.csv"
PRIM_BOXES_FILE = "prim_boxes.csv"

SCENARIO_CONFIG = {
    "NI": {"name": "No Incentive", "color": "viridis"},
    "SI": {"name": "Services Incentive", "color": "viridis"},
    "EI": {"name": "Economic Incentive", "color": "viridis"}
}

PRIM_BOX_COLOR = "yellow"
PRIM_BOX_LINEWIDTH = 2


def load_csv(filename: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load CSV file from data directory.
    
    Args:
        filename: Name of CSV file to load
        data_dir: Directory containing data files
        
    Returns:
        DataFrame containing loaded data
        
    Raises:
        FileNotFoundError: If file does not exist
    """
    filepath = data_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


def prepare_heatmap_data(
    df: pd.DataFrame, 
    scenario: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare heatmap data for a specific scenario.
    
    Args:
        df: DataFrame containing heatmap grid data
        scenario: Scenario identifier (NI, SI, EI)
        
    Returns:
        Tuple of (trust_bins, income_bins, adoption_rate_grid)
    """
    scenario_data = df[df["scenario"] == scenario].copy()
    
    # Extract unique bins
    trust_bins = sorted(scenario_data["trust_bin"].unique())
    income_bins = sorted(scenario_data["income_bin"].unique())
    
    # Create 2D grid for adoption rates
    adoption_grid = np.zeros((len(income_bins), len(trust_bins)))
    
    for _, row in scenario_data.iterrows():
        trust_idx = trust_bins.index(row["trust_bin"])
        income_idx = income_bins.index(row["income_bin"])
        adoption_grid[income_idx, trust_idx] = row["adoption_rate"]
    
    return np.array(trust_bins), np.array(income_bins), adoption_grid


def create_prim_box_patch(
    box: pd.Series,
    trust_bins: np.ndarray,
    income_bins: np.ndarray
) -> mpatches.Rectangle:
    """
    Create matplotlib Rectangle patch for PRIM box overlay.
    
    Args:
        box: Series containing PRIM box boundaries
        trust_bins: Array of trust bin centers
        income_bins: Array of income bin centers
        
    Returns:
        Rectangle patch object
    """
    # Find indices closest to box boundaries
    trust_min_idx = np.searchsorted(trust_bins, box["trust_min"])
    trust_max_idx = np.searchsorted(trust_bins, box["trust_max"])
    income_min_idx = np.searchsorted(income_bins, box["income_min"])
    income_max_idx = np.searchsorted(income_bins, box["income_max"])
    
    # Calculate rectangle parameters
    x = trust_min_idx - 0.5
    y = income_min_idx - 0.5
    width = trust_max_idx - trust_min_idx
    height = income_max_idx - income_min_idx
    
    return mpatches.Rectangle(
        (x, y), width, height,
        fill=False,
        edgecolor=PRIM_BOX_COLOR,
        linewidth=PRIM_BOX_LINEWIDTH,
        linestyle="-",
        zorder=10
    )


def plot_single_heatmap(
    ax: plt.Axes,
    trust_bins: np.ndarray,
    income_bins: np.ndarray,
    adoption_grid: np.ndarray,
    scenario: str,
    prim_box: Optional[pd.Series] = None,
    cmap: str = "viridis"
) -> plt.cm.ScalarMappable:
    """
    Plot a single heatmap on given axes.
    
    Args:
        ax: Matplotlib axes object
        trust_bins: Array of trust bin centers
        income_bins: Array of income bin centers
        adoption_grid: 2D array of adoption rates
        scenario: Scenario identifier
        prim_box: Optional PRIM box data for overlay
        cmap: Colormap name
        
    Returns:
        ScalarMappable object for colorbar
    """
    im = ax.imshow(
        adoption_grid,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[
            -0.5, len(trust_bins) - 0.5,
            -0.5, len(income_bins) - 0.5
        ]
    )
    
    # Add PRIM box if provided
    if prim_box is not None:
        patch = create_prim_box_patch(prim_box, trust_bins, income_bins)
        ax.add_patch(patch)
    
    # Configure axes
    ax.set_title(SCENARIO_CONFIG[scenario]["name"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Trust", fontsize=10)
    ax.set_ylabel("Income", fontsize=10)
    
    # Set tick labels
    trust_ticks = np.linspace(0, len(trust_bins) - 1, 5).astype(int)
    income_ticks = np.linspace(0, len(income_bins) - 1, 5).astype(int)
    
    ax.set_xticks(trust_ticks)
    ax.set_xticklabels([f"{trust_bins[i]:.2f}" for i in trust_ticks])
    ax.set_yticks(income_ticks)
    ax.set_yticklabels([f"{income_bins[i]:.0f}" for i in income_ticks])
    
    return im


def plot_adoption_heatmaps(
    heatmap_file: str = HEATMAP_FILE,
    prim_boxes_file: str = PRIM_BOXES_FILE,
    scenarios: List[str] = ["NI", "SI", "EI"],
    figsize: Tuple[int, int] = (15, 4),
    output_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Generate heatmaps of adoption rate across scenarios with PRIM boxes.
    
    Args:
        heatmap_file: Name of heatmap data CSV file
        prim_boxes_file: Name of PRIM boxes CSV file
        scenarios: List of scenario identifiers to plot
        figsize: Figure size (width, height)
        output_path: Optional path to save figure
        dpi: Resolution for saved figure
        
    Returns:
        Matplotlib Figure object
    """
    # Load data
    heatmap_df = load_csv(heatmap_file)
    prim_boxes_df = load_csv(prim_boxes_file)
    
    # Create figure
    fig, axes = plt.subplots(1, len(scenarios), figsize=figsize)
    if len(scenarios) == 1:
        axes = [axes]
    
    # Plot each scenario
    for ax, scenario in zip(axes, scenarios):
        # Prepare heatmap data
        trust_bins, income_bins, adoption_grid = prepare_heatmap_data(
            heatmap_df, scenario
        )
        
        # Get PRIM box for scenario
        prim_box = prim_boxes_df[prim_boxes_df["scenario"] == scenario]
        prim_box = prim_box.iloc[0] if not prim_box.empty else None
        
        # Plot heatmap
        im = plot_single_heatmap(
            ax, trust_bins, income_bins, adoption_grid,
            scenario, prim_box, SCENARIO_CONFIG[scenario]["color"]
        )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Adoption Rate", fontsize=10)
    
    # Add overall title
    fig.suptitle(
        "Adoption Rate as a Function of Trust and Income",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )
    
    # Adjust layout (rect parameter avoids suptitle overlap)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def main():
    """Main execution function for standalone usage."""
    fig = plot_adoption_heatmaps(
        output_path=Path("data/results/figures/figure1_heatmaps.png")
    )
    plt.show()


if __name__ == "__main__":
    main()