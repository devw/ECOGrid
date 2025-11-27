"""
Main controller script for the Adoption Heatmap Generator.
This script orchestrates data loading, processing, statistical analysis, 
and plotting, utilizing modularized utility functions.
"""

from pathlib import Path

# External Libraries needed for control and plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle # Needed for the HeatmapPlotter class definition

# Import constants from the centralized config
from ._config.settings import *

# Import I/O functions from the centralized data module
from ._data_io.csv_reader import load_csv, load_metadata
from ._processors.data_utils import scenario_grid
from ._processors.stats_utils import compute_pairwise_significance
from .plotting import HeatmapPlotter

# --- LOGIC SECTIONS (To be moved in next steps) ---

# Note: The following sections (GRID/PIVOT HELPERS, STATISTICAL ANALYSIS, PLOTTING CLASS)
# are left here temporarily but should be moved to:
# - _processors/data_utils.py
# - _processors/stats_utils.py
# - plotting.py
# as discussed in the refactoring plan.

from itertools import combinations
import numpy as np
import pandas as pd

# --- CONTROLLER ---

def plot_all(output: Path) -> plt.Figure:
    """Orchestrates the entire heatmap generation process."""
    try:
        # Data loading now uses the imported functions from _data_io/csv_reader.py
        df = load_csv(HEATMAP_FILE)
        prim_df = load_csv(PRIM_BOXES_FILE)
        meta = load_metadata()
    except FileNotFoundError as e:
        print(f"Data loading error: {e}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    sig_path = output.parent / "statistical_significance_results.csv"
    if not sig_path.exists():
        sig = compute_pairwise_significance(df)
        sig.to_csv(sig_path, index=False)
    else:
        sig = pd.read_csv(sig_path)

    grids = {}
    for s in SCENARIOS:
        grids[s] = scenario_grid(df, s)
        p = prim_df[prim_df["scenario"] == s]
        grids[s]['prim'] = p.iloc[0] if not p.empty else None

    # Increased figure size for robust spacing
    fig = plt.figure(figsize=(10, 16)) 
    plotter = HeatmapPlotter()

    n_reps = grids[next(iter(SCENARIOS))]['n_replications']
    
    # Suptitle split into two lines
    suptitle_text = f"Adoption Rate Heatmaps: Trust vs Income Statistical Analysis\n(N={n_reps} Monte Carlo replications)"
    fig.suptitle(suptitle_text, y=SUPTITLE_Y, fontsize=FONTSIZE_TITLE, fontweight='bold')

    # Compact horizontal colorbar positioned lower
    cbar_ax = fig.add_axes([0.10, COLORBAR_Y_POS, 0.80, 0.01]) 

    last_im = None
    fig.subplots_adjust(hspace=0.3) 
    for i, (code, title) in enumerate(SCENARIOS.items(), 1):
        ax = fig.add_subplot(len(SCENARIOS), 1, i)
        last_im = plotter.plot_single(ax, grids[code], title, grids[code]['prim'], show_ci=(i == 1), meta=meta)

    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')
    
    legend_patches = [mpatches.Patch(facecolor='none', edgecolor=PRIM_COLOR, linewidth=PRIM_WIDTH, linestyle='-', label='PRIM Box (High-Adoption Region)')]
    
    # 2. Position the Colorbar label in a new axis below the Colorbar
    cbar_label_ax = fig.add_axes([0.10, CBAR_LABEL_Y_POS, 0.80, 0.01])
    cbar_label_ax.axis('off')
    cbar_label_ax.text(0.5, 0.5, COLORBAR_LABEL, ha='center', va='center', fontsize=FONTSIZE_CBAR_LABEL)
    
    # 3. Position the Legend in a new axis below the Colorbar label
    legend_ax = fig.add_axes([0.10, LEGEND_Y_POS, 0.80, 0.01])
    legend_ax.axis('off') 
    # Move the legend slightly to the right to balance the space
    legend_ax.legend(handles=legend_patches, loc='center right', fontsize=FONTSIZE_TEXT_SMALL + 1, framealpha=0.8, handlelength=2.5, borderpad=0.2)

    # Statistical note in the footer
    fig.text(0.5, 0.005, "Error bars (subplot 1) show 95% confidence intervals ($\text{CI}$). See console for statistical significance tests.", ha='center', fontsize=FONTSIZE_TEXT_SMALL, style='italic', color='gray')

    # Adjusted top margin
    fig.tight_layout(rect=[0.12, 0.015, 1, 0.85]) 

    # Improvement: Set resolution to 300 DPI
    fig.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✔ Heatmap saved to: {output.resolve()}")

    return fig


if __name__ == "__main__":
    plot_all(OUTPUT_PATH)