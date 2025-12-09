"""CSV writing utilities for PRIM trajectory data."""

import csv
from pathlib import Path
from typing import List
from src.data.schemas import PRIMTrajectoryReplicationSchema, PRIMTrajectoryEnhancedSchema  # (gli schemas necessari)


def save_prim_trajectory_summary(data: List[PRIMTrajectoryEnhancedSchema], filepath: Path):
    """Save PRIM trajectory summary with statistics."""
    import csv
    
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


def save_prim_trajectory_replications(data: List[PRIMTrajectoryReplicationSchema], filepath: Path):
    """Save all PRIM trajectory replications."""
    import csv
    
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