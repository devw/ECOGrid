"""
Main controller script for the Adoption Heatmap Generator.
Orchestrates data loading, processing, statistical analysis, and plotting.
"""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import constants (styles, colors)
from ._config.settings import *

# Import functional modules
from ._data_io.csv_reader import load_csv, load_metadata
from ._processors.data_utils import scenario_grid
from ._processors.stats_utils import compute_pairwise_significance
from .plotting import HeatmapPlotter


def plot_all(data_dir: Path, output: Path) -> plt.Figure:
    """Orchestrates the entire heatmap generation process: load, analyze, and plot."""
    
    # 1. DATA LOADING
    try:
        df = load_csv(data_dir / HEATMAP_FILE)
        prim_df = load_csv(data_dir / PRIM_BOXES_FILE)
        meta = load_metadata(data_dir / METADATA_FILE)
    except FileNotFoundError as e:
        print(f"Data loading error: {e}")
        return

    # 2. STATISTICAL ANALYSIS & CACHING
    output.parent.mkdir(parents=True, exist_ok=True)
    sig_path = data_dir / "statistical_significance_results.csv"
    
    if sig_path.exists():
        sig = load_csv(sig_path)
    else:
        sig = compute_pairwise_significance(df)
        sig.to_csv(sig_path, index=False)

    # 3. DATA PROCESSING & GRID PREPARATION
    grids = {}
    for scenario_code in SCENARIOS:
        grids[scenario_code] = scenario_grid(df, scenario_code)
        p = prim_df[prim_df["scenario"] == scenario_code]
        grids[scenario_code]['prim'] = p.iloc[0] if not p.empty else None

    # 4. PLOTTING SETUP
    fig = plt.figure(figsize=(10, 16)) 
    plotter = HeatmapPlotter()
    n_reps = grids[next(iter(SCENARIOS))]['n_replications']
    
    suptitle_text = (
        f"Adoption Rate Heatmaps: Trust vs Income Statistical Analysis\n"
        f"(N={n_reps} Monte Carlo replications)"
    )
    fig.suptitle(suptitle_text, y=SUPTITLE_Y, fontsize=FONTSIZE_TITLE, fontweight='bold')

    cbar_ax = fig.add_axes([0.10, COLORBAR_Y_POS, 0.80, 0.01]) 
    fig.subplots_adjust(hspace=0.3) 

    # 5. PLOT GENERATION LOOP
    last_im = None
    for i, (scenario_code, title) in enumerate(SCENARIOS.items(), 1):
        ax = fig.add_subplot(len(SCENARIOS), 1, i)
        last_im = plotter.plot_single(
            ax, 
            grids[scenario_code], 
            title, 
            grids[scenario_code]['prim'], 
            show_ci=(i == 1), 
            meta=meta
        )

    # 6. POST-PLOTTING & ANNOTATIONS
    fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')
    
    legend_patches = [
        mpatches.Patch(
            facecolor='none', 
            edgecolor=PRIM_COLOR, 
            linewidth=PRIM_WIDTH, 
            linestyle='-', 
            label='PRIM Box (High-Adoption Region)'
        )
    ]
    
    cbar_label_ax = fig.add_axes([0.10, CBAR_LABEL_Y_POS, 0.80, 0.01])
    cbar_label_ax.axis('off')
    cbar_label_ax.text(0.5, 0.5, COLORBAR_LABEL, ha='center', va='center', fontsize=FONTSIZE_CBAR_LABEL)
    
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

    fig.text(
        0.5, 
        0.005, 
        "Error bars (subplot 1) show 95% confidence intervals (CI). See console for statistical significance tests.", 
        ha='center', 
        fontsize=FONTSIZE_TEXT_SMALL, 
        style='italic', 
        color='gray'
    )

    fig.tight_layout(rect=[0.12, 0.015, 1, 0.85]) 
    fig.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✔ Heatmap saved to: {output.resolve()}")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate adoption heatmaps for any data pipeline."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/montecarlo",
        help="Directory containing input CSV files (MonteCarlo, MESA, etc.)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/adoption_heatmaps.png",
        help="Path to save the heatmap figure"
    )
    args = parser.parse_args()

    plot_all(Path(args.data_dir), Path(args.output))
