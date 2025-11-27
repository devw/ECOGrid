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
from scipy import stats

# --- GRID / PIVOT HELPERS ---

def _pivot_grid(df: pd.DataFrame, value_col: str) -> np.ndarray:
    """Pivot and return a 2D numpy array aligned by income_bin (rows) and trust_bin (cols)."""
    table = (
        df.pivot_table(index="income_bin", columns="trust_bin", values=value_col, fill_value=0)
          .sort_index()
          .sort_index(axis=1)
    )
    return table.values


def scenario_grid(df: pd.DataFrame, scenario: str) -> dict:
    """Extract grid arrays and basic metadata for a scenario."""
    sub = df[df["scenario"] == scenario].copy()
    if sub.empty:
        raise ValueError(f"No data for scenario {scenario}")

    trust = np.sort(sub["trust_bin"].unique())
    income = np.sort(sub["income_bin"].unique())

    return {
        "trust": trust,
        "income": income,
        "adoption": _pivot_grid(sub, "adoption_rate"),
        "std_dev": _pivot_grid(sub, "std_dev"),
        "ci_lower": _pivot_grid(sub, "ci_lower"),
        "ci_upper": _pivot_grid(sub, "ci_upper"),
        "n_replications": int(sub["n_replications"].iloc[0]) if "n_replications" in sub.columns else None,
    }

# --- STATISTICAL ANALYSIS ---

def compute_pairwise_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Welch t-test p-values per grid cell between scenarios."""
    rows = []
    for (t, y), group in df.groupby(["trust_bin", "income_bin"]):
        for s1, s2 in combinations(SCENARIOS.keys(), 2):
            a = group[group["scenario"] == s1]
            b = group[group["scenario"] == s2]
            if a.empty or b.empty or len(a) == 0 or len(b) == 0:
                continue

            mean1, std1, n1 = a[["adoption_rate", "std_dev", "n_replications"]].values[0]
            mean2, std2, n2 = b[["adoption_rate", "std_dev", "n_replications"]].values[0]

            se2 = (std1**2 / n1) + (std2**2 / n2)
            if se2 <= 0:
                pval = 1.0
            else:
                tstat = (mean1 - mean2) / np.sqrt(se2)
                denom = (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1)
                df_ = (se2 ** 2) / denom if denom > 0 else (n1 + n2 - 2)
                pval = 2 * (1 - stats.t.cdf(abs(tstat), df_))

            rows.append({
                "trust_bin": t,
                "income_bin": y,
                "scenario_A": s1,
                "scenario_B": s2,
                "mean_diff": mean1 - mean2,
                "p_value": pval,
                "significant": pval < P_VALUE_THRESHOLD,
            })

    return pd.DataFrame(rows)

# --- PLOTTING CLASS ---
class HeatmapPlotter:
    """Handle plotting of a single heatmap with overlays and PRIM box annotation."""

    @staticmethod
    def _prim_patch(box: pd.Series, trust: np.ndarray, income: np.ndarray) -> Rectangle:
        idx = lambda arr, val: np.searchsorted(arr, val)
        x0 = idx(trust, box["trust_min"]) - 0.5
        y0 = idx(income, box["income_min"]) - 0.5
        w = idx(trust, box["trust_max"]) - idx(trust, box["trust_min"])
        h = idx(income, box["income_max"]) - idx(income, box["income_min"])
        # Improvement: Changed linestyle to "-" and used updated PRIM_COLOR/PRIM_WIDTH
        return Rectangle((x0, y0), w, h, fill=False, edgecolor=PRIM_COLOR, linewidth=PRIM_WIDTH, linestyle="-", zorder=10)

    @staticmethod
    def _prim_label(box: pd.Series) -> str:
        return f"PRIM Box: Coverage={box['coverage']:.0%}, Density={box['density']:.0%}, Lift={box['lift']:.1f}"

    @staticmethod
    def _add_ci_overlay(ax: plt.Axes, trust: np.ndarray, income: np.ndarray, grid: dict, step: int):
        xs = np.arange(0, len(trust), step)
        ys = np.arange(0, len(income), step)
        for i in ys:
            for j in xs:
                m = grid['adoption'][i, j]
                lo = grid['ci_lower'][i, j]
                hi = grid['ci_upper'][i, j]
                ax.plot([j, j], [i + (lo - m), i + (hi - m)], color='white', alpha=0.5, linewidth=1.0, zorder=6)


    def plot_single(self, ax: plt.Axes, grid: dict, title: str, prim_box: pd.Series | None, show_ci: bool, meta: dict) -> plt.Axes:
        trust, income = grid['trust'], grid['income']
        im = ax.imshow(grid['adoption'], origin='lower', aspect='auto', cmap=CMAP, vmin=0, vmax=1,
                       extent=[-0.5, len(trust) - 0.5, -0.5, len(income) - 0.5])

        if prim_box is not None:
            ax.add_patch(self._prim_patch(prim_box, trust, income))
            # Improvement: Removed alpha for better text contrast
            ax.text(0.98, 0.02, self._prim_label(prim_box), transform=ax.transAxes, fontsize=FONTSIZE_TEXT_SMALL, color=PRIM_COLOR,
                    ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='black', alpha=1.0), zorder=11)

        if show_ci:
            self._add_ci_overlay(ax, trust, income, grid, CI_DOWNSAMPLE)

        # Improvement: Uniformed statistical notation in the title (mu/CI)
        ax.set_title(
            f"{title} (N={grid['n_replications']}, $\\mu \\pm 95\\% \\text{{ CI}}$)",
            fontsize=FONTSIZE_SUBTITLE,
            fontweight='bold'
        )
        
        # Y-Axis label split into two lines
        y_label = meta.get('income', {}).get('interpretation', 'Income (0→100)')
        y_label = y_label.replace('(0=lowest, 100=highest)', '\n(0=lowest, 100=highest)')
        ax.set_xlabel(meta.get('trust', {}).get('interpretation', 'Trust (0→1)'), fontsize=FONTSIZE_AXES_LABEL)
        ax.set_ylabel(y_label, fontsize=FONTSIZE_AXES_LABEL)

        xt = np.linspace(0, len(trust) - 1, 5).astype(int)
        yt = np.linspace(0, len(income) - 1, 5).astype(int)
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_xticklabels([f"{trust[i]:.2f}" for i in xt], fontsize=FONTSIZE_AXES_TICKS)
        ax.set_yticklabels([f"{income[i]:.0f}" for i in yt], fontsize=FONTSIZE_AXES_TICKS)

        # Improvement: Uniformed statistical notation for standard deviation ($\sigma$)
        ax.text(0.02, 0.98, f"Avg $\\sigma$={np.mean(grid['std_dev']):.3f}", transform=ax.transAxes, fontsize=FONTSIZE_TEXT_SMALL, color='white',
                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5), zorder=11)

        return im

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