"""
Main controller script for the Adoption Heatmap Generator.
Orchestrates data loading, processing, statistical analysis, and plotting.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.utils.cli_parser import base_parser, safe_run

from ._config.settings import SCENARIOS, FONTSIZE_TITLE, SUPTITLE_Y, COLORBAR_Y_POS, PRIM_COLOR, PRIM_WIDTH, CBAR_LABEL_Y_POS, LEGEND_Y_POS, COLORBAR_LABEL, FONTSIZE_CBAR_LABEL, FONTSIZE_TEXT_SMALL
from ._processors.data_utils import scenario_grid
from ._processors.stats_utils import compute_pairwise_significance
from .plotting import HeatmapPlotter
from ._utils.file_utils import load_csv_or_fail
from ._processors.report_generator import print_analysis_report


def print_analysis_report(grids: dict, scenarios: dict):
    """Print textual analysis of adoption rates and PRIM boxes."""
    
    print("\n" + "="*80)
    print("📊 ADOPTION RATE & PRIM BOX ANALYSIS REPORT")
    print("="*80 + "\n")
    
    for code, title in scenarios.items():
        grid = grids[code]
        prim = grid.get('prim')
        
        print(f"{'─'*80}")
        print(f"📌 {title.upper()}")
        print(f"{'─'*80}")
        
        # Adoption rate statistics
        adoption_data = grid['adoption'].flatten()
        print(f"\n  📈 Adoption Rate Statistics:")
        print(f"     • Average (α):  {grid['avg_adoption']:.1%}")
        print(f"     • Minimum:      {adoption_data.min():.1%}")
        print(f"     • Maximum:      {adoption_data.max():.1%}")
        print(f"     • Std Dev (σ):  {adoption_data.std():.3f}")
        print(f"     • Replications: {grid['n_replications']:,}")
        
        # PRIM Box analysis
        if prim is not None:
            print(f"\n  🎯 PRIM Box (High-Adoption Region):")
            print(f"     • Coverage:  {prim['coverage']:.1%} of population")
            print(f"     • Density:   {prim['density']:.1%} of high-adoption cases")
            print(f"     • Lift:      {prim['lift']:.2f}x above average")
            
            # Calculate estimated adoption in PRIM box
            estimated_adoption = grid['avg_adoption'] * prim['lift']
            print(f"     • Est. adoption in box: ~{estimated_adoption:.1%}")
            
            # Box boundaries
            print(f"\n  📦 Box Boundaries:")
            print(f"     • Trust:  [{prim['trust_min']:.3f}, {prim['trust_max']:.3f}]")
            print(f"     • Income: [{prim['income_min']:.1f}, {prim['income_max']:.1f}]")
            
            # Interpretation
            print(f"\n  💡 Interpretation:")
            if prim['coverage'] < 0.15:
                reach = "Very limited reach (elite only)"
            elif prim['coverage'] < 0.30:
                reach = "Limited reach (excludes majority)"
            elif prim['coverage'] < 0.50:
                reach = "Moderate reach (covers minority)"
            else:
                reach = "Good reach (covers majority)"
            print(f"     • Population reach: {reach}")
            
            if prim['lift'] > 2.0:
                effectiveness = "Highly effective in target region"
            elif prim['lift'] > 1.5:
                effectiveness = "Moderately effective in target region"
            else:
                effectiveness = "Modest effectiveness in target region"
            print(f"     • Effectiveness: {effectiveness}")
        else:
            print(f"\n  ⚠️  No PRIM box identified (low variance in adoption)")
        
        print()  # Empty line between scenarios
    
    print("="*80 + "\n")


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

    # 3.5 PRINT ANALYSIS REPORT
    print_analysis_report(grids, SCENARIOS)


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

def main():
    args = base_parser(defaults={
        "data_dir": Path("data/montecarlo"),
        "output": Path("/tmp/adoption_heatmap.png"),
    }).parse_args()

    plot_all(args.output, args.data_dir)

if __name__ == "__main__":
    safe_run(main)