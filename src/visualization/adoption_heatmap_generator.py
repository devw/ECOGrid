"""
Main controller script for the Adoption Heatmap Generator.
Orchestrates data loading, processing, statistical analysis, and plotting.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import constants (paths, styles, colors)
from ._config.settings import *

# Import functional modules
from ._data_io.csv_reader import load_csv, load_metadata
from ._processors.data_utils import scenario_grid
from ._processors.stats_utils import compute_pairwise_significance
from .plotting import HeatmapPlotter

# --- CONTROLLER ---

def plot_all(output: Path) -> plt.Figure:
    """Orchestrates the entire heatmap generation process: load, analyze, and plot."""
    
    # 1. DATA LOADING
    try:
        df = load_csv(HEATMAP_FILE)
        prim_df = load_csv(PRIM_BOXES_FILE)
        meta = load_metadata()
    except FileNotFoundError as e:
        print(f"Data loading error: {e}")
        return

    # 2. STATISTICAL ANALYSIS & CACHING
    output.parent.mkdir(parents=True, exist_ok=True)
    sig_path = output.parent / "statistical_significance_results.csv"
    
    if sig_path.exists():
        sig = pd.read_csv(sig_path)
    else:
        sig = compute_pairwise_significance(df)
        sig.to_csv(sig_path, index=False)

    # 3. DATA PROCESSING & GRID PREPARATION
    grids = {}
    for s in SCENARIOS:
        grids[s] = scenario_grid(df, s)
        # Retrieve PRIM box data for the current scenario
        p = prim_df[prim_df["scenario"] == s]
        grids[s]['prim'] = p.iloc[0] if not p.empty else None

    # 4. PLOTTING SETUP (Figure, Suptitle, Colorbar Axis)
    fig = plt.figure(figsize=(10, 16)) 
    plotter = HeatmapPlotter()

    n_reps = grids[next(iter(SCENARIOS))]['n_replications']
    
    suptitle_text = f"Adoption Rate Heatmaps: Trust vs Income Statistical Analysis\n(N={n_reps} Monte Carlo replications)"
    fig.suptitle(suptitle_text, y=SUPTITLE_Y, fontsize=FONTSIZE_TITLE, fontweight='bold')

    # Define axis for the horizontal colorbar
    cbar_ax = fig.add_axes([0.10, COLORBAR_Y_POS, 0.80, 0.01]) 
    fig.subplots_adjust(hspace=0.3) 

    # 5. PLOT GENERATION LOOP
    last_im = None
    for i, (code, title) in enumerate(SCENARIOS.items(), 1):
        ax = fig.add_subplot(len(SCENARIOS), 1, i)
        # Show CI only for the first subplot
        last_im = plotter.plot_single(
            ax, 
            grids[code], 
            title, 
            grids[code]['prim'], 
            show_ci=(i == 1), 
            meta=meta
        )

    # 6. POST-PLOTTING & ANNOTATIONS
    
    # Add colorbar
    fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')
    
    # Create patch for the legend
    legend_patches = [
        mpatches.Patch(
            facecolor='none', 
            edgecolor=PRIM_COLOR, 
            linewidth=PRIM_WIDTH, 
            linestyle='-', 
            label='PRIM Box (High-Adoption Region)'
        )
    ]
    
    # Position the Colorbar label
    cbar_label_ax = fig.add_axes([0.10, CBAR_LABEL_Y_POS, 0.80, 0.01])
    cbar_label_ax.axis('off')
    cbar_label_ax.text(0.5, 0.5, COLORBAR_LABEL, ha='center', va='center', fontsize=FONTSIZE_CBAR_LABEL)
    
    # Position the Legend
    legend_ax = fig.add_axes([0.10, LEGEND_Y_POS, 0.80, 0.01])
    legend_ax.axis('off') 
    legend_ax.legend(
        handles=legend_patches, 
        loc='center right', 
        fontsize=FONTSIZE_TEXT_SMALL + 1, 
        framealpha=0.8, 
        handlelength=2.5, 
        borderpad=0.2
    )

    # Add statistical note in the footer
    fig.text(
        0.5, 
        0.005, 
        "Error bars (subplot 1) show 95% confidence intervals ($\text{CI}$). See console for statistical significance tests.", 
        ha='center', 
        fontsize=FONTSIZE_TEXT_SMALL, 
        style='italic', 
        color='gray'
    )

    # Save figure
    fig.tight_layout(rect=[0.12, 0.015, 1, 0.85]) 
    fig.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✔ Heatmap saved to: {output.resolve()}")

    return fig


if __name__ == "__main__":
    plot_all(OUTPUT_PATH)