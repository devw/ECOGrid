# src/utils/plot_style.py

SCENARIO_COLORS = {
    "NI": "blue",
    "EI": "green",
    "SI": "orange"
}

ERRORBAR_STYLE = {
    "linewidth": 2,
    "marker": "o",
    "markersize": 4,
    "capsize": 6,
    "elinewidth": 1.5,
    "zorder": 2
}

SELECTED_BOX_STYLE = {
    "marker": "*",
    "s": 500,
    "edgecolor": "gray",
    "linewidth": 0.1,
    "zorder": 3
}

ANNOTATION_STYLE = {
    "text_offset": (5, 15),
    "fontsize": 12,
    "arrowprops": dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
        linewidth=0.7
    )
}
