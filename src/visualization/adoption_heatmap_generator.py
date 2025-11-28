"""
Main controller script for the Adoption Heatmap Generator.
Orchestrates data loading, processing, statistical analysis, and plotting.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ._config.settings import SCENARIOS, FONTSIZE_TITLE, SUPTITLE_Y, COLORBAR_Y_POS, PRIM_COLOR, PRIM_WIDTH, CBAR_LABEL_Y_POS, LEGEND_Y_POS, COLORBAR_LABEL, FONTSIZE_CBAR_LABEL, FONTSIZE_TEXT_SMALL
from ._processors.data_utils import scenario_grid
from ._processors.stats_utils import compute_pairwise_significance
from .plotting import HeatmapPlotter
from ._utils.file_utils import load_csv_or_fail

def plot_all(output: Path, data_dir: Path):
    """Orchestrates heatmap generation: load data, analyze, and plot."""

    # 1. DATA LOADING
    heatmap_file = data_dir / "heatmap_grid.csv"
    prim_file = data_dir / "prim_boxes.csv"
    metadata_file = data_dir / "scale_metadata.json"
    
    df = load_csv_or_fail(heatmap_file)
    prim_df = load_csv_or_fail(prim_file)
    meta = {}  

    # 2. STATISTICAL ANALYSIS & CACHING
    output.parent.mkdir(parents=True, exist_ok=True)
    sig_path = output.parent / "statistical_significance_results.csv"
    if sig_path.exists():
        print(f"📄 Loading CSV: {sig_path}")
        sig = load_csv_or_fail(sig_path)
    else:
        sig = compute_pairwise_significance(df)
        sig.to_csv(sig_path, index=False)
        print(f"✔ Statistical significance saved: {sig_path}")

    # 3. DATA PROCESSING & GRID PREPARATION
    grids = {s: scenario_grid(df, s) for s in SCENARIOS}
    for s in SCENARIOS:
        p = prim_df[prim_df["scenario"] == s]
        grids[s]['prim'] = p.iloc[0] if not p.empty else None

    # 4. PLOTTING
    fig = plt.figure(figsize=(10, 16))
    plotter = HeatmapPlotter()
    n_reps = grids[next(iter(SCENARIOS))]['n_replications']

    fig.suptitle(
        f"Adoption Rate Heatmaps: Trust vs Income Statistical Analysis\n(N={n_reps} Monte Carlo replications)",
        y=SUPTITLE_Y, fontsize=FONTSIZE_TITLE, fontweight='bold'
    )
    cbar_ax = fig.add_axes([0.10, COLORBAR_Y_POS, 0.80, 0.01])
    fig.subplots_adjust(hspace=0.3)

    last_im = None
    for i, (code, title) in enumerate(SCENARIOS.items(), 1):
        ax = fig.add_subplot(len(SCENARIOS), 1, i)
        last_im = plotter.plot_single(
            ax,
            grids[code],
            title,
            grids[code]['prim'],
            show_ci=(i == 1),
            meta=meta
        )

    # Add colorbar
    fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')

    # Add legend
    legend_ax = fig.add_axes([0.10, LEGEND_Y_POS, 0.80, 0.01])
    legend_ax.axis('off')
    legend_ax.legend(
        handles=[mpatches.Patch(facecolor='none', edgecolor=PRIM_COLOR, linewidth=PRIM_WIDTH, linestyle='-', label='PRIM Box (High-Adoption Region)')],
        loc='center right',
        fontsize=FONTSIZE_TEXT_SMALL + 1,
        framealpha=0.8,
        handlelength=2.5,
        borderpad=0.2
    )

    # Colorbar label
    cbar_label_ax = fig.add_axes([0.10, CBAR_LABEL_Y_POS, 0.80, 0.01])
    cbar_label_ax.axis('off')
    cbar_label_ax.text(0.5, 0.5, COLORBAR_LABEL, ha='center', va='center', fontsize=FONTSIZE_CBAR_LABEL)

    # Footer note
    fig.text(
        0.5,
        0.005,
        "Error bars (subplot 1) show 95% confidence intervals ($\\text{CI}$).",
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
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", type=Path, default=Path("data/montecarlo"), help="Directory containing CSV/JSON files")
        parser.add_argument("--output", type=Path, default=Path("/tmp/adoption_heatmap.png"), help="Path to save the output figure")
        args = parser.parse_args()
        plot_all(args.output, args.data_dir)
    except FileNotFoundError as e:
        import sys
        print(f"❌ {e}")
        sys.exit(1)  # Exit immediately without traceback