"""
Adoption Heatmap Generator (Enhanced with Statistical Metrics)

Generates heatmaps of adoption rate vs. trust & income for multiple scenarios.
Includes:
- Statistical significance testing between scenarios
- Confidence intervals as error bars overlays
- N replications clearly indicated
- Standardized and documented scales
- Quantitative PRIM box thresholds

Output:
    /tmp/adoption_heatmaps_enhanced.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy import stats
import json

DATA_DIR = Path("data/dummy")
HEATMAP_FILE = "heatmap_grid.csv"
PRIM_BOXES_FILE = "prim_boxes.csv"
METADATA_FILE = "scale_metadata.json"

SCENARIOS = {
    "NI": "No Incentive",
    "SI": "Services Incentive",
    "EI": "Economic Incentive"
}

CMAP = "viridis"
PRIM_COLOR = "yellow"
PRIM_WIDTH = 2.5


# ------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------

def load_csv(name: str) -> pd.DataFrame:
    """Load CSV file from data directory."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def load_metadata() -> dict:
    """Load scale metadata JSON."""
    path = DATA_DIR / METADATA_FILE
    if not path.exists():
        print(f"⚠️  Warning: {METADATA_FILE} not found, using defaults")
        return {}
    with open(path, 'r') as f:
        return json.load(f)


# ------------------------------------------------------------
# Grid Preparation
# ------------------------------------------------------------

def scenario_grid(df: pd.DataFrame, s: str):
    """Extract 2D grid data for a scenario with statistics."""
    data = df[df["scenario"] == s].copy()
    trust = sorted(data["trust_bin"].unique())
    income = sorted(data["income_bin"].unique())

    # Pivot for mean adoption rate
    grid = data.pivot_table(
        index="income_bin",
        columns="trust_bin",
        values="adoption_rate",
        fill_value=0
    ).reindex(index=income, columns=trust).values

    # Pivot for std_dev
    std_grid = data.pivot_table(
        index="income_bin",
        columns="trust_bin",
        values="std_dev",
        fill_value=0
    ).reindex(index=income, columns=trust).values

    # Pivot for confidence intervals
    ci_lower_grid = data.pivot_table(
        index="income_bin",
        columns="trust_bin",
        values="ci_lower",
        fill_value=0
    ).reindex(index=income, columns=trust).values

    ci_upper_grid = data.pivot_table(
        index="income_bin",
        columns="trust_bin",
        values="ci_upper",
        fill_value=0
    ).reindex(index=income, columns=trust).values

    # Get n_replications (should be constant)
    n_reps = int(data["n_replications"].iloc[0]) if "n_replications" in data.columns else None

    return {
        "trust": np.array(trust),
        "income": np.array(income),
        "adoption": grid,
        "std_dev": std_grid,
        "ci_lower": ci_lower_grid,
        "ci_upper": ci_upper_grid,
        "n_replications": n_reps
    }


# ------------------------------------------------------------
# Statistical Significance Testing
# ------------------------------------------------------------

def compute_pairwise_significance(heatmap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical significance between scenarios using t-tests.
    
    Returns DataFrame with columns:
    - trust_bin, income_bin, scenario_A, scenario_B, p_value, significant
    """
    from itertools import combinations
    
    results = []
    scenario_list = list(SCENARIOS.keys())
    
    # Group by grid cell
    for (trust, income), group in heatmap_df.groupby(['trust_bin', 'income_bin']):
        # For each pair of scenarios
        for s1, s2 in combinations(scenario_list, 2):
            data_s1 = group[group['scenario'] == s1]
            data_s2 = group[group['scenario'] == s2]
            
            if len(data_s1) == 0 or len(data_s2) == 0:
                continue
            
            # Extract statistics
            mean1 = data_s1['adoption_rate'].values[0]
            std1 = data_s1['std_dev'].values[0]
            n1 = data_s1['n_replications'].values[0]
            
            mean2 = data_s2['adoption_rate'].values[0]
            std2 = data_s2['std_dev'].values[0]
            n2 = data_s2['n_replications'].values[0]
            
            # Two-sample t-test using summary statistics
            # Pooled standard error
            se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
            t_stat = (mean1 - mean2) / se if se > 0 else 0
            
            # Degrees of freedom (Welch's approximation)
            df = ((std1**2/n1 + std2**2/n2)**2) / \
                 ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            results.append({
                'trust_bin': trust,
                'income_bin': income,
                'scenario_A': s1,
                'scenario_B': s2,
                'mean_diff': mean1 - mean2,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    return pd.DataFrame(results)


def count_significant_differences(sig_df: pd.DataFrame, scenario_pair: tuple) -> int:
    """Count how many grid cells show significant differences between two scenarios."""
    mask = ((sig_df['scenario_A'] == scenario_pair[0]) & 
            (sig_df['scenario_B'] == scenario_pair[1])) | \
           ((sig_df['scenario_A'] == scenario_pair[1]) & 
            (sig_df['scenario_B'] == scenario_pair[0]))
    
    return sig_df[mask & sig_df['significant']].shape[0]


# ------------------------------------------------------------
# PRIM Box Visualization
# ------------------------------------------------------------

def prim_box_patch(box, trust, income):
    """Create a rectangle highlighting a PRIM box with quantitative labels."""
    def idx(arr, val): return np.searchsorted(arr, val)

    x0 = idx(trust, box["trust_min"]) - 0.5
    y0 = idx(income, box["income_min"]) - 0.5
    w = idx(trust, box["trust_max"]) - idx(trust, box["trust_min"])
    h = idx(income, box["income_max"]) - idx(income, box["income_min"])

    return Rectangle(
        (x0, y0), w, h,
        fill=False, edgecolor=PRIM_COLOR, linewidth=PRIM_WIDTH, 
        linestyle='--', zorder=10
    )


def format_prim_label(box) -> str:
    """Format PRIM box statistics as a readable label."""
    return (f"PRIM Box: Coverage={box['coverage']:.0%}, "
            f"Density={box['density']:.0%}, Lift={box['lift']:.1f}")


# ------------------------------------------------------------
# Error Bar Overlay
# ------------------------------------------------------------

def add_uncertainty_overlay(ax, trust, income, grid_data, downsample=4):
    """
    Add error bars showing confidence intervals as overlays.
    
    Args:
        ax: Matplotlib axis
        trust, income: Bin centers
        grid_data: Dictionary with adoption, ci_lower, ci_upper
        downsample: Show every Nth point to avoid clutter
    """
    adoption = grid_data['adoption']
    ci_lower = grid_data['ci_lower']
    ci_upper = grid_data['ci_upper']
    
    # Downsample for readability
    x_idx = np.arange(0, len(trust), downsample)
    y_idx = np.arange(0, len(income), downsample)
    
    for i in y_idx:
        for j in x_idx:
            mean = adoption[i, j]
            lower = ci_lower[i, j]
            upper = ci_upper[i, j]
            
            # Draw vertical error bar
            ax.plot([j, j], [i + (lower - mean) * 10, i + (upper - mean) * 10],
                   color='white', alpha=0.4, linewidth=0.8, zorder=5)


# ------------------------------------------------------------
# Enhanced Plotting
# ------------------------------------------------------------

def plot_heatmap(ax, grid_data, title, prim, show_errorbar=True):
    """
    Plot a single scenario heatmap with all enhancements.
    
    Args:
        ax: Matplotlib axis
        grid_data: Dictionary with trust, income, adoption, statistics
        title: Subplot title
        prim: PRIM box data (or None)
        show_errorbar: Whether to overlay error bars
    """
    trust = grid_data['trust']
    income = grid_data['income']
    adoption = grid_data['adoption']
    std_dev = grid_data['std_dev']
    n_reps = grid_data['n_replications']
    
    # Main heatmap
    im = ax.imshow(
        adoption, origin="lower", aspect="auto", cmap=CMAP,
        vmin=0, vmax=1,
        extent=[-0.5, len(trust)-0.5, -0.5, len(income)-0.5]
    )

    # Add PRIM box if present
    if prim is not None:
        patch = prim_box_patch(prim, trust, income)
        ax.add_patch(patch)
        
        # Add PRIM statistics as text annotation
        prim_text = format_prim_label(prim)
        ax.text(0.98, 0.02, prim_text,
               transform=ax.transAxes,
               fontsize=8, color=PRIM_COLOR,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6),
               ha='right', va='bottom', zorder=11)

    # Add uncertainty overlay (if enabled)
    if show_errorbar and std_dev is not None:
        add_uncertainty_overlay(ax, trust, income, grid_data, downsample=5)

    # Title with N replications
    title_with_stats = f"{title}\n(N={n_reps} replications, Mean ± 95% CI)"
    ax.set_title(title_with_stats, fontweight="bold", fontsize=11)
    
    ax.set_xlabel("Trust (normalized, 0=low → 1=high)", fontsize=9)
    ax.set_ylabel("Income (percentile, 0=low → 100=high)", fontsize=9)

    # Better tick labels with units from metadata
    xt = np.linspace(0, len(trust)-1, 5).astype(int)
    yt = np.linspace(0, len(income)-1, 5).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{trust[i]:.2f}" for i in xt], fontsize=8)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{income[i]:.0f}" for i in yt], fontsize=8)

    # Add mean std annotation
    mean_std = np.mean(std_dev)
    ax.text(0.02, 0.98, f"Avg σ={mean_std:.3f}",
           transform=ax.transAxes,
           fontsize=8, color='white',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
           ha='left', va='top', zorder=11)

    return im


def plot_all(output: Path):
    """Generate complete enhanced heatmap figure."""
    # Load data
    heatmap = load_csv(HEATMAP_FILE)
    prim_boxes = load_csv(PRIM_BOXES_FILE)
    metadata = load_metadata()
    
    # Compute statistical significance
    print("🔬 Computing statistical significance tests...")
    sig_results = compute_pairwise_significance(heatmap)
    
    # Print summary of significance testing
    print("\n📊 Statistical Significance Summary:")
    print("=" * 60)
    for (s1, s2) in [("NI", "SI"), ("NI", "EI"), ("SI", "EI")]:
        n_sig = count_significant_differences(sig_results, (s1, s2))
        total_cells = len(heatmap[heatmap['scenario'] == s1])
        pct = (n_sig / total_cells * 100) if total_cells > 0 else 0
        print(f"  {SCENARIOS[s1]} vs {SCENARIOS[s2]}: "
              f"{n_sig}/{total_cells} cells significant ({pct:.1f}%)")
    print("=" * 60 + "\n")
    
    # Prepare grid data for all scenarios
    grid_data = {}
    for s in SCENARIOS:
        grid_data[s] = scenario_grid(heatmap, s)
        grid_data[s]['prim'] = (
            prim_boxes[prim_boxes["scenario"] == s].iloc[0]
            if not prim_boxes[prim_boxes["scenario"] == s].empty else None
        )
    
    # Create figure with enhanced layout
    fig = plt.figure(figsize=(8, 12))
    
    # Main title with metadata info
    n_reps = grid_data[next(iter(SCENARIOS))]['n_replications']
    if metadata:
        trust_info = metadata.get('trust', {}).get('interpretation', 'Trust score')
        income_info = metadata.get('income', {}).get('interpretation', 'Income percentile')
        suptitle = (f"Adoption Rate Heatmaps: Trust vs Income\n"
                   f"Statistical Analysis (N={n_reps} Monte Carlo replications)")
    else:
        suptitle = f"Adoption Rate Heatmaps (N={n_reps} replications)"
    
    fig.suptitle(suptitle, fontweight="bold", fontsize=13, y=0.98)
    
    # Create subplots
    axes = []
    for i, (s, title) in enumerate(SCENARIOS.items(), 1):
        ax = fig.add_subplot(len(SCENARIOS), 1, i)
        axes.append(ax)
        
        im = plot_heatmap(ax, grid_data[s], title, grid_data[s]['prim'], 
                         show_errorbar=(i == 1))  # Show error bars only on first
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.15, 0.96, 0.7, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Adoption Rate (0=none → 1=full adoption)", fontsize=9)
    
    # Add legend for PRIM boxes
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor=PRIM_COLOR, 
                      linewidth=PRIM_WIDTH, linestyle='--',
                      label='PRIM Box (High-Adoption Region)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', 
                  fontsize=8, framealpha=0.8)
    
    # Add statistical significance note
    fig.text(0.5, 0.01, 
            "Error bars (subplot 1) show 95% confidence intervals. "
            "See console for statistical significance tests.",
            ha='center', fontsize=8, style='italic', color='gray')
    
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # Save figure
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"✅ Figure saved to: {output}")
    
    # Also save significance results
    sig_output = output.parent / "statistical_significance_results.csv"
    sig_results.to_csv(sig_output, index=False)
    print(f"✅ Significance tests saved to: {sig_output}")
    
    return fig


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    """Main execution function."""
    output = Path("/tmp/adoption_heatmaps_enhanced.png")
    
    print("=" * 70)
    print("🎨 Enhanced Adoption Heatmap Generator")
    print("=" * 70)
    print("Features:")
    print("  ✓ Statistical significance testing (t-tests)")
    print("  ✓ Confidence intervals (95% CI)")
    print("  ✓ N replications clearly displayed")
    print("  ✓ Standardized scales with units")
    print("  ✓ Quantitative PRIM box thresholds")
    print("=" * 70 + "\n")
    
    plot_all(output)
    
    print("\n" + "=" * 70)
    print("✅ Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()