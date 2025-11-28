from pathlib import Path
import pandas as pd

def load_csv_or_fail(csv_path: Path | str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    
    if not csv_path.parent.exists():
        print(f"❌ Directory not found: {csv_path.parent}. Please check the path.")
        raise FileNotFoundError(f"Directory does not exist: {csv_path.parent}")
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}. Please make sure the required CSV is present.")
        raise FileNotFoundError(f"File does not exist: {csv_path}")

    print(f"📄 Loading CSV: {csv_path}")
    return pd.read_csv(csv_path)



def load_metadata_or_fail(json_path: Path) -> dict:
    """
    Load a metadata JSON file or stop execution if the file does not exist or is invalid.

    Parameters:
    - json_path: Path to the JSON file.

    Returns:
    - dict: Loaded metadata.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"❌ Metadata file not found: {json_path}. Please check the path and try again.")
    
    import json
    try:
        with open(json_path, "r") as f:
            print(f"📄 Loading metadata JSON: {json_path}")
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ JSON decode error in {json_path}: {e}")
