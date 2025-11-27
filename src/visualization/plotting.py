import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Importa costanti dai settings, necessarie per lo styling
from ._config.settings import (
    CMAP, PRIM_COLOR, PRIM_WIDTH, 
    CI_DOWNSAMPLE, FONTSIZE_TEXT_SMALL, 
    FONTSIZE_SUBTITLE, FONTSIZE_AXES_LABEL, 
    FONTSIZE_AXES_TICKS
) 

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
        # Changed linestyle to "-" and used updated PRIM_COLOR/PRIM_WIDTH
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
            # Removed alpha for better text contrast
            ax.text(0.98, 0.02, self._prim_label(prim_box), transform=ax.transAxes, fontsize=FONTSIZE_TEXT_SMALL, color=PRIM_COLOR,
                    ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='black', alpha=1.0), zorder=11)

        if show_ci:
            self._add_ci_overlay(ax, trust, income, grid, CI_DOWNSAMPLE)

        # Uniformed statistical notation in the title (mu/CI)
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

        # Uniformed statistical notation for standard deviation ($\sigma$)
        ax.text(0.02, 0.98, f"Avg $\\sigma$={np.mean(grid['std_dev']):.3f}", transform=ax.transAxes, fontsize=FONTSIZE_TEXT_SMALL, color='white',
                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5), zorder=11)

        return im