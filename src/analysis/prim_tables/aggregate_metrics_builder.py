"""
Build aggregate metrics table from heatmap grid data.

This module calculates scenario-level metrics including:
- Mean adoption rate across all bins
- Adoption rates by income/trust brackets
- Income gap analysis
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, Any

from src.utils.file_utils import load_csv_or_fail


# Income thresholds (terzili)
LOW_INCOME_THRESHOLD = 33.33
HIGH_INCOME_THRESHOLD = 66.67

# Trust threshold (from PRIM analysis)
HIGH_TRUST_THRESHOLD = 0.6357  # Closest bin center to 0.65


def load_metadata(data_dir: Path) -> Dict[str, Any]:
    """Load scale metadata if available."""
    metadata_path = data_dir / "scale_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def calculate_weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str = 'n_samples') -> float:
    """Calculate weighted mean of a column."""
    total_weight = df[weight_col].sum()
    if total_weight == 0:
        return 0.0
    return (df[value_col] * df[weight_col]).sum() / total_weight


def calculate_weighted_std(df: pd.DataFrame, value_col: str, weight_col: str = 'n_samples') -> float:
    """Calculate weighted standard deviation."""
    mean = calculate_weighted_mean(df, value_col, weight_col)
    total_weight = df[weight_col].sum()
    if total_weight == 0:
        return 0.0
    variance = ((df[value_col] - mean) ** 2 * df[weight_col]).sum() / total_weight
    return np.sqrt(variance)


def compute_scenario_metrics(scenario_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute aggregate metrics for a single scenario.
    
    Args:
        scenario_df: DataFrame filtered for one scenario
        
    Returns:
        Dictionary with computed metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['mean_adoption_rate'] = calculate_weighted_mean(scenario_df, 'adoption_rate')
    metrics['std_adoption_rate'] = calculate_weighted_std(scenario_df, 'adoption_rate')
    
    # High-trust adoption (trust_bin >= HIGH_TRUST_THRESHOLD)
    high_trust = scenario_df[scenario_df['trust_bin'] >= HIGH_TRUST_THRESHOLD]
    if len(high_trust) > 0:
        metrics['high_trust_adoption'] = calculate_weighted_mean(high_trust, 'adoption_rate')
    else:
        metrics['high_trust_adoption'] = np.nan
    
    # High-income adoption (income_bin >= HIGH_INCOME_THRESHOLD)
    high_income = scenario_df[scenario_df['income_bin'] >= HIGH_INCOME_THRESHOLD]
    if len(high_income) > 0:
        metrics['high_income_adoption'] = calculate_weighted_mean(high_income, 'adoption_rate')
    else:
        metrics['high_income_adoption'] = np.nan
    
    # Low-income adoption (income_bin < LOW_INCOME_THRESHOLD)
    low_income = scenario_df[scenario_df['income_bin'] < LOW_INCOME_THRESHOLD]
    if len(low_income) > 0:
        metrics['low_income_adoption'] = calculate_weighted_mean(low_income, 'adoption_rate')
    else:
        metrics['low_income_adoption'] = np.nan
    
    # Income gap (high - low)
    if not np.isnan(metrics['high_income_adoption']) and not np.isnan(metrics['low_income_adoption']):
        metrics['income_gap'] = metrics['high_income_adoption'] - metrics['low_income_adoption']
    else:
        metrics['income_gap'] = np.nan
    
    # Number of bins (for reference)
    metrics['n_bins'] = len(scenario_df)
    
    return metrics


def build_aggregate_metrics_table(data_dir: Path) -> pd.DataFrame:
    """
    Build aggregate metrics table from heatmap grid data.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        DataFrame with aggregate metrics by scenario
    """
    # Load data
    heatmap_path = data_dir / "heatmap_grid.csv"
    df = load_csv_or_fail(heatmap_path)
    
    # Load metadata (optional)
    metadata = load_metadata(data_dir)
    
    # Compute metrics for each scenario
    scenarios = df['scenario'].unique()
    results = []
    
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario]
        metrics = compute_scenario_metrics(scenario_df)
        metrics['scenario'] = scenario
        results.append(metrics)
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        'scenario',
        'mean_adoption_rate',
        'std_adoption_rate',
        'high_trust_adoption',
        'high_income_adoption',
        'low_income_adoption',
        'income_gap',
        'n_bins'
    ]
    
    result_df = result_df[column_order]
    
    return result_df