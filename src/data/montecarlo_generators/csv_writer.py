"""
CSV writing utilities for Monte Carlo simulation data.

This module handles all CSV and JSON file output operations for
the agent-based energy transition model.
"""

import csv
import json
from pathlib import Path
from typing import List

from src.data.schemas import (
    PRIMTrajectoryEnhancedSchema,
    PRIMTrajectoryReplicationSchema
)
from src.data.csv_utils import schemas_to_csv
from .config import GeneratorConfig
from .metadata_generator import generate_scale_metadata


def save_prim_trajectory_summary(
    data: List[PRIMTrajectoryEnhancedSchema], 
    filepath: Path
) -> None:
    """Save PRIM trajectory summary with statistics."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scenario', 'iteration', 
            'coverage_mean', 'coverage_std', 'coverage_ci_lower', 'coverage_ci_upper',
            'density_mean', 'density_std', 'density_ci_lower', 'density_ci_upper',
            'n_agents_mean', 'is_selected', 'n_replications'
        ])
        for item in data:
            writer.writerow([
                item.scenario.value,
                item.iteration,
                item.coverage_mean,
                item.coverage_std,
                item.coverage_ci_lower,
                item.coverage_ci_upper,
                item.density_mean,
                item.density_std,
                item.density_ci_lower,
                item.density_ci_upper,
                item.n_agents_mean,
                item.is_selected,
                item.n_replications
            ])


def save_prim_trajectory_replications(
    data: List[PRIMTrajectoryReplicationSchema], 
    filepath: Path
) -> None:
    """Save all PRIM trajectory replications."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scenario', 'iteration', 'replication_id',
            'coverage', 'density', 'n_agents', 'is_selected'
        ])
        for item in data:
            writer.writerow([
                item.scenario.value,
                item.iteration,
                item.replication_id,
                item.coverage,
                item.density,
                item.n_agents,
                item.is_selected
            ])


def save_all_data(all_data: dict, output_dir: Path, config: GeneratorConfig) -> None:
    """
    Save all generated data to CSV files.
    
    Args:
        all_data: Dictionary with all scenario data
        output_dir: Directory to save CSV files
        config: Generator configuration (for metadata)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving data to {output_dir.absolute()}")
    
    # Aggregate data across scenarios
    all_heatmaps = []
    all_heatmap_reps = []
    all_prim_boxes = []
    all_trajectories_summary = []
    all_trajectories_reps = []
    all_profiles = []
    
    for scenario_name, data in all_data.items():
        all_heatmaps.extend(data['heatmap_grid'])
        all_heatmap_reps.extend(data['heatmap_replications'])
        all_prim_boxes.extend(data['prim_boxes'])
        all_trajectories_summary.extend(data['prim_trajectory_summary'])
        all_trajectories_reps.extend(data['prim_trajectory_replications'])
        all_profiles.extend(data['demographic_profiles'])
    
    # Save aggregated heatmap (with statistics)
    print("  ├─ Saving heatmap_grid.csv (with CI)...")
    schemas_to_csv(all_heatmaps, output_dir / "heatmap_grid.csv")
    
    # Save disaggregated replications
    print("  ├─ Saving heatmap_replications.csv (all runs)...")
    schemas_to_csv(all_heatmap_reps, output_dir / "heatmap_replications.csv")
    
    # Save other files
    schemas_to_csv(all_prim_boxes, output_dir / "prim_boxes.csv")
    
    # Save PRIM trajectory summary
    print("  ├─ Saving prim_trajectory_summary.csv (with CI)...")
    save_prim_trajectory_summary(all_trajectories_summary, output_dir / "prim_trajectory_summary.csv")
    
    # Save PRIM trajectory replications
    print("  ├─ Saving prim_trajectory_raw.csv (all runs)...")
    save_prim_trajectory_replications(all_trajectories_reps, output_dir / "prim_trajectory_raw.csv")
    
    schemas_to_csv(all_profiles, output_dir / "demographic_profiles.csv")
    
    # Save scale metadata
    print("  └─ Saving scale_metadata.json...")
    metadata = generate_scale_metadata(config)
    with open(output_dir / "scale_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ All data saved successfully!")