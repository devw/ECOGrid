from pathlib import Path
import json
import pandas as pd
from .._config.settings import HEATMAP_FILE, PRIM_BOXES_FILE, METADATA_FILE, DATA_DIR
from src.utils.file_utils import ensure_file_or_fallback

def load_csv(file_name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load a CSV file. If missing, delegate fallback handling to file_utils.
    """
    path = Path(data_dir) / file_name
    path = ensure_file_or_fallback(path)  # centralized fallback + logging

    return pd.read_csv(path)


def load_metadata(data_dir: Path = DATA_DIR) -> dict:
    """
    Load metadata JSON. If missing, delegate fallback handling to file_utils.
    """
    path = Path(data_dir) / METADATA_FILE
    path = ensure_file_or_fallback(path, fallback_type="metadata")
    
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ JSON decode error in {path}. Returning empty metadata...")
        return {}
