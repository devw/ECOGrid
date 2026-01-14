"""
Rebuild demographic_profiles.csv from prim_boxes.csv with correct lift values.

This script recalculates lift values using the correct formula:
    lift = density / population_mean_adoption

Where population_mean_adoption is calculated from heatmap_grid.csv as a
population-weighted average across all trust-income bins.

Usage:
    python -m src.scripts.rebuild_demographic_profiles \
      --data-dir data/montecarlo_calibrated_fixed \
      --n-agents 5000
"""
from pathlib import Path
import pandas as pd
import shutil

import logging

from src.utils.file_utils import load_csv_or_fail
from src.utils.cli_parser import base_parser, safe_run

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def calculate_population_mean(heatmap_df: pd.DataFrame, scenario: str) -> float:
    """
    Calculate population-weighted mean adoption rate for a scenario.
    
    Args:
        heatmap_df: DataFrame from heatmap_grid.csv with columns:
                   [scenario, adoption_rate, n_samples, ...]
        scenario: Scenario name (NI, SI, EI)
    
    Returns:
        float: Population-weighted mean adoption rate
        
    Raises:
        ValueError: If no data found for scenario
    """
    scenario_data = heatmap_df[heatmap_df['scenario'] == scenario].copy()
    
    if len(scenario_data) == 0:
        raise ValueError(f"No data found for scenario: {scenario}")
    
    # Population-weighted mean: Σ(adoption_rate × n_samples) / Σ(n_samples)
    total_samples = scenario_data['n_samples'].sum()
    weighted_adoption = (scenario_data['adoption_rate'] * scenario_data['n_samples']).sum()
    
    return weighted_adoption / total_samples


def generate_segment_name(row: pd.Series) -> str:
    """
    Generate descriptive segment name from PRIM box bounds.
    
    Args:
        row: DataFrame row with columns [scenario, trust_min, income_min, ...]
    
    Returns:
        str: Human-readable segment description
    """
    scenario = row['scenario']
    trust_min = row['trust_min']
    income_min = row['income_min']
    
    if scenario == 'NI':
        return "Baseline Population (No Segmentation)"
    elif scenario == 'SI':
        return f"High Trust Community (Trust ≥ {trust_min:.2f})"
    elif scenario == 'EI':
        return f"High Trust + Mid-High Income (Trust ≥ {trust_min:.2f}, Income ≥ {income_min:.0f})"
    else:
        # Fallback for unknown scenarios
        return f"Segment {scenario}"


def rebuild_demographic_profiles(data_dir: Path, n_agents: int) -> pd.DataFrame:
    """
    Rebuild demographic_profiles.csv from prim_boxes.csv with correct lift.
    
    This function:
    1. Loads prim_boxes.csv (contains PRIM segment definitions)
    2. Loads heatmap_grid.csv (contains population-level adoption rates)
    3. Calculates correct lift = density / population_mean
    4. Rebuilds demographic_profiles.csv with corrected values
    
    Args:
        data_dir: Directory containing CSV files
        n_agents: Total number of agents per replication
        
    Returns:
        pd.DataFrame: Rebuilt demographic profiles table
        
    Raises:
        FileNotFoundError: If required CSV files are missing
    """
    logger.info("=" * 80)
    logger.info("REBUILDING DEMOGRAPHIC_PROFILES.CSV WITH CORRECT LIFT")
    logger.info("=" * 80)
    
    # Define file paths
    prim_boxes_path = data_dir / "prim_boxes.csv"
    heatmap_path = data_dir / "heatmap_grid.csv"
    output_path = data_dir / "demographic_profiles.csv"
    
    # Load source data
    logger.info(f"Loading {prim_boxes_path.name}")
    prim_df = load_csv_or_fail(prim_boxes_path)
    
    logger.info(f"Loading {heatmap_path.name}")
    heatmap_df = load_csv_or_fail(heatmap_path)
    
    # Backup original file if it exists
    if output_path.exists():
        backup_path = output_path.with_suffix('.csv.backup')
        shutil.copy(output_path, backup_path)
        logger.info(f"Backup created: {backup_path.name}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("CALCULATING CORRECT LIFT VALUES")
    logger.info("=" * 80)
    
    # Rebuild demographic profiles with correct lift
    records = []
    
    for _, row in prim_df.iterrows():
        scenario = row['scenario']
        density = row['density']
        coverage = row['coverage']
        
        # Calculate population mean for this scenario
        pop_mean = calculate_population_mean(heatmap_df, scenario)
        
        # Calculate CORRECT lift
        lift_correct = density / pop_mean if pop_mean > 0 else 0.0
        
        # Calculate n_agents_segment from coverage
        n_segment = int(coverage * n_agents)
        
        # Generate descriptive segment name
        segment_name = generate_segment_name(row)
        
        # Build record
        record = {
            'scenario': scenario,
            'segment_name': segment_name,
            'trust_min': row['trust_min'],
            'trust_max': row['trust_max'],
            'income_min': row['income_min'],
            'income_max': row['income_max'],
            'coverage': coverage,
            'density': density,
            'lift': lift_correct,  # ← CORRECTED LIFT
            'n_agents_total': n_agents,
            'n_agents_segment': n_segment
        }
        
        records.append(record)
        
        # Log calculations
        logger.info(f"\n{scenario} ({segment_name}):")
        logger.info(f"  Density:          {density:.6f} ({density*100:.2f}%)")
        logger.info(f"  Coverage:         {coverage:.6f} ({coverage*100:.2f}%)")
        logger.info(f"  Population Mean:  {pop_mean:.6f} ({pop_mean*100:.2f}%)")
        logger.info(f"  Lift (correct):   {lift_correct:.6f}")
        logger.info(f"  n_segment:        {n_segment}")
    
    # Create DataFrame
    demographic_df = pd.DataFrame(records)
    
    # Save to CSV
    demographic_df.to_csv(output_path, index=False)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ DEMOGRAPHIC_PROFILES.CSV REBUILT SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Output: {output_path}")
    logger.info(f"Columns: {list(demographic_df.columns)}")
    logger.info(f"Rows: {len(demographic_df)}")
    
    # Display summary table
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 80)
    summary_cols = ['scenario', 'coverage', 'density', 'lift']
    logger.info(f"\n{demographic_df[summary_cols].to_string(index=False)}")
    
    return demographic_df


def main():
    """Main entry point for script execution."""
    # Parse arguments using project's base parser
    # Note: base_parser already includes --data-dir and --n-agents
    parser = base_parser(defaults={"output": Path(".")})
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    # Get n_agents from args or use default
    n_agents = getattr(args, 'n_agents', 5000)
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Number of agents: {n_agents}")
    logger.info("")
    
    # Rebuild demographic profiles
    rebuild_demographic_profiles(data_dir, n_agents)
    
    logger.info("")
    logger.info("✅ Script completed successfully!")


if __name__ == "__main__":
    safe_run(main)