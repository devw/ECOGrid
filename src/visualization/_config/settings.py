from pathlib import Path

# --- CONFIGURATION ---
# Dummy paths for execution context
DATA_DIR = Path("data/montecarlo")
HEATMAP_FILE = "heatmap_grid.csv"
PRIM_BOXES_FILE = "prim_boxes.csv"
METADATA_FILE = "scale_metadata.json"
# Updated output path to reflect final version
OUTPUT_PATH = Path("/tmp/adoption_heatmaps.png") 

SCENARIOS = {
    "NI": "No Incentive",
    "SI": "Services Incentive",
    "EI": "Economic Incentive",
}

CMAP = "viridis"
PRIM_COLOR = "red" 
PRIM_WIDTH = 3.5   
P_VALUE_THRESHOLD = 0.05
CI_DOWNSAMPLE = 5
SUPTITLE_Y = 0.98 
COLORBAR_LABEL = "Adoption Rate (0=none → 1=full adoption)"

# FONT SIZES (Increased by +2pt)
FONTSIZE_TITLE = 18 
FONTSIZE_SUBTITLE = 16
FONTSIZE_AXES_LABEL = 14 
FONTSIZE_AXES_TICKS = 12 
FONTSIZE_TEXT_SMALL = 12
FONTSIZE_CBAR_LABEL = 14 

# OPTIMIZED POSITIONS (More aggressive separation)
COLORBAR_Y_POS = 0.92 
CBAR_LABEL_Y_POS = 0.89
LEGEND_Y_POS = 0.86