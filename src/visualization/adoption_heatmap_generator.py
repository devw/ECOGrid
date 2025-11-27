"""
Adoption Heatmap Generator — Final Vertical Text Separation

Improvements applied:
- Explicitly separated the Colorbar's label and the Legend's label vertically.
- The Colorbar label is now placed *below* the colorbar itself.
- The Legend is placed *below* the Colorbar label, resolving the final overlap.
"""

from pathlib import Path
import json
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy import stats

# --- CONFIGURATION ---
# Dummy paths for execution context
DATA_DIR = Path("data/dummy")
HEATMAP_FILE = "heatmap_grid.csv"
PRIM_BOXES_FILE = "prim_boxes.csv"
METADATA_FILE = "scale_metadata.json"
# Updated output path to reflect final version
OUTPUT_PATH = Path("/tmp/adoption_heatmaps_final_final_optimized.png") 

SCENARIOS = {
    "NI": "No Incentive",
    "SI": "Services Incentive",
    "EI": "Economic Incentive",
}

CMAP = "viridis"
PRIM_COLOR = "yellow"
PRIM_WIDTH = 2.5
P_VALUE_THRESHOLD = 0.05
CI_DOWNSAMPLE = 5
SUPTITLE_Y = 0.98 
COLORBAR_LABEL = "Adoption Rate (0=none → 1=full adoption)"

# FONT SIZES
FONTSIZE_TITLE = 16 
FONTSIZE_SUBTITLE = 14
FONTSIZE_AXES_LABEL = 12 
FONTSIZE_AXES_TICKS = 10 
FONTSIZE_TEXT_SMALL = 10
FONTSIZE_CBAR_LABEL = 12 

# OPTIMIZED POSITIONS (More aggressive separation)
COLORBAR_Y_POS = 0.92 
CBAR_LABEL_Y_POS = 0.89  # New position for the Colorbar's label
LEGEND_Y_POS = 0.86     # New position for the Legend (below CBAR_LABEL)

# --- IO UTILITIES ---

def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        # Creating dummy data if files are not found
        if name == HEATMAP_FILE:
            num_bins = 10
            np.random.seed(42)
            trust_bins = np.linspace(0.03, 0.97, num_bins)
            income_bins = np.linspace(2, 98, num_bins)
            data = []
            for scenario in ['NI', 'SI', 'EI']:
                for t_bin in trust_bins:
                    for i_bin in income_bins:
                        base_rate = (t_bin + i_bin / 100) / 2
                        if scenario == 'SI': base_rate += 0.1 * (t_bin > 0.6)
                        elif scenario == 'EI': base_rate += 0.1 * (i_bin > 60)
                        adoption = np.clip(base_rate + np.random.randn() * 0.05, 0, 1)
                        std_dev = np.random.rand() * 0.04 + 0.01
                        data.append({
                            'scenario': scenario,
                            'trust_bin': t_bin,
                            'income_bin': i_bin,
                            'adoption_rate': adoption,
                            'std_dev': std_dev,
                            'ci_lower': np.clip(adoption - 1.96 * std_dev / np.sqrt(100), 0, 1),
                            'ci_upper': np.clip(adoption + 1.96 * std_dev / np.sqrt(100), 0, 1),
                            'n_replications': 1000,
                        })
            return pd.DataFrame(data)
        elif name == PRIM_BOXES_FILE:
            return pd.DataFrame({
                'scenario': ['NI', 'SI', 'EI'],
                'trust_min': [0.03, 0.5, 0.6],
                'trust_max': [0.97, 0.97, 0.97],
                'income_min': [2, 2, 60],
                'income_max': [98, 98, 98],
                'coverage': [1.0, 0.5, 0.3],
                'density': [0.5, 0.9, 0.8],
                'lift': [1.0, 1.8, 1.6],
            })
        elif name == METADATA_FILE:
            return {
                "trust": {"interpretation": "Agent trust propensity score (0=no trust, 1=full trust)"},
                "income": {"interpretation": "Income percentile in population (0=lowest, 100=highest)"}
            }
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def load_metadata() -> dict:
    path = DATA_DIR / METADATA_FILE
    if not path.exists():
        return {
            "trust": {"interpretation": "Agent trust propensity score (0=no trust, 1=full trust)"},
            "income": {"interpretation": "Income percentile in population (0=lowest, 100=highest)"}
        }
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

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
        return Rectangle((x0, y0), w, h, fill=False, edgecolor=PRIM_COLOR, linewidth=PRIM_WIDTH, linestyle="--", zorder=10)

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
            ax.text(0.98, 0.02, self._prim_label(prim_box), transform=ax.transAxes, fontsize=FONTSIZE_TEXT_SMALL, color=PRIM_COLOR,
                    ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6), zorder=11)

        if show_ci:
            self._add_ci_overlay(ax, trust, income, grid, CI_DOWNSAMPLE)

        ax.set_title(
            f"{title} (N={grid['n_replications']} replications, Mean +/- 95% CI)",
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

        ax.text(0.02, 0.98, f"Avg $\\sigma$={np.mean(grid['std_dev']):.3f}", transform=ax.transAxes, fontsize=FONTSIZE_TEXT_SMALL, color='white',
                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5), zorder=11)

        return im

# --- CONTROLLER ---

def plot_all(output: Path) -> plt.Figure:
    try:
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

    # INCREASED FIGURE SIZE FOR ROBUST SPACING
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
    
    # 1. Rimuovi l'etichetta di default dalla colorbar
    # cbar.set_label(COLORBAR_LABEL, fontsize=FONTSIZE_CBAR_LABEL) # Commented out

    legend_patches = [mpatches.Patch(facecolor='none', edgecolor=PRIM_COLOR, linewidth=PRIM_WIDTH, linestyle='--', label='PRIM Box (High-Adoption Region)')]
    
    # 2. Posiziona l'etichetta della Colorbar in un nuovo asse sotto la Colorbar
    cbar_label_ax = fig.add_axes([0.10, CBAR_LABEL_Y_POS, 0.80, 0.01])
    cbar_label_ax.axis('off')
    cbar_label_ax.text(0.5, 0.5, COLORBAR_LABEL, ha='center', va='center', fontsize=FONTSIZE_CBAR_LABEL)
    
    # 3. Posiziona la Legenda in un nuovo asse sotto l'etichetta della Colorbar
    legend_ax = fig.add_axes([0.10, LEGEND_Y_POS, 0.80, 0.01])
    legend_ax.axis('off') 
    # Spostiamo la legenda leggermente a destra per bilanciare lo spazio
    legend_ax.legend(handles=legend_patches, loc='center right', fontsize=FONTSIZE_TEXT_SMALL + 1, framealpha=0.8, handlelength=2.5, borderpad=0.2)

    fig.text(0.5, 0.005, "Error bars (subplot 1) show 95% confidence intervals. See console for statistical significance tests.", ha='center', fontsize=FONTSIZE_TEXT_SMALL, style='italic', color='gray')

    # Adjusted top margin
    fig.tight_layout(rect=[0.12, 0.015, 1, 0.85]) 

    fig.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✔ Heatmap saved to: {output.resolve()}")

    return fig


if __name__ == "__main__":
    plot_all(OUTPUT_PATH)