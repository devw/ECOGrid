"""
PRIM Trajectory Generation - Data-Driven from YAML Config.
Zero hardcoded values, fully configurable.
"""
from typing import List, Tuple
from dataclasses import asdict
import numpy as np
import pandas as pd

from src.data.schemas import (
    ScenarioType,
    PRIMTrajectoryReplicationSchema,
    PRIMTrajectoryEnhancedSchema
)
from src.utils.montecarlo_stats import aggregate_replications
from .scenario_config import get_prim_trajectory_config, get_global_config


def _calculate_trajectory_arrays(
    config,
    n_iterations: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculate base trajectory arrays from config.
    
    Returns:
        (base_coverages, base_densities, selected_iteration)
    """
    base_coverages = np.linspace(
        config.coverage_start,
        config.coverage_end,
        n_iterations
    )
    
    if config.density_exponent is not None:
        base_densities = (
            config.density_base + 
            config.density_coefficient * (1 - base_coverages) ** config.density_exponent
        )
    else:
        base_densities = np.full(n_iterations, config.density_base)
    
    selected_iteration = n_iterations + config.selected_iteration_offset
    
    return base_coverages, base_densities, selected_iteration


def generate_prim_trajectory_replications(
    scenario: ScenarioType,
    n_iterations: int,
    n_replications: int,
    noise_std: float,
    random_state: np.random.RandomState
) -> List[PRIMTrajectoryReplicationSchema]:
    """
    Generate PRIM peeling trajectory data with replications.
    Uses configuration from YAML - zero hardcoded values.
    """
    config = get_prim_trajectory_config(scenario)
    global_config = get_global_config()
    
    base_coverages, base_densities, selected_iteration = _calculate_trajectory_arrays(
        config, n_iterations
    )
    
    all_replications = []
    
    for rep_id in range(n_replications):
        for i in range(n_iterations):
            coverage_noise = random_state.normal(
                0, 
                noise_std * global_config.coverage_noise_scale
            )
            density_noise = random_state.normal(
                0, 
                noise_std * global_config.density_noise_scale
            )
            
            coverage = np.clip(base_coverages[i] + coverage_noise, 0.0, 1.0)
            density = np.clip(base_densities[i] + density_noise, 0.0, 1.0)
            
            all_replications.append(PRIMTrajectoryReplicationSchema(
                scenario=scenario,
                iteration=i,
                replication_id=rep_id,
                coverage=coverage,
                density=density,
                n_agents=int(global_config.n_agents_total * coverage),
                is_selected=(i == selected_iteration)
            ))
    
    return all_replications


def aggregate_prim_trajectory_replications(
    replications: List[PRIMTrajectoryReplicationSchema]
) -> List[PRIMTrajectoryEnhancedSchema]:
    """
    Aggregate PRIM trajectory replications into summary statistics.
    Uses stats_utils to compute confidence intervals.
    """
    global_config = get_global_config()
    
    df = pd.DataFrame([asdict(r) for r in replications])
    df = df.rename(columns={'replication_id': 'run_id'})
    
    aggregated = aggregate_replications(
        df,
        group_cols=['scenario', 'iteration'],
        value_cols=['coverage', 'density'],
        confidence=0.95,
        ci_method='parametric'
    )
    
    trajectory_enhanced = []
    
    # Create lookup for is_selected (same for all replications in group)
    original_df = pd.DataFrame([asdict(r) for r in replications])
    selected_lookup = original_df.groupby(['scenario', 'iteration'])['is_selected'].first()
    
    for _, row in aggregated.iterrows():
        key = (row['scenario'], row['iteration'])
        is_selected = selected_lookup[key]
        
        trajectory_enhanced.append(PRIMTrajectoryEnhancedSchema(
            scenario=ScenarioType(row['scenario']),
            iteration=int(row['iteration']),
            coverage_mean=row['coverage_mean'],
            coverage_std=row['coverage_std'],
            coverage_ci_lower=row['coverage_ci_lower'],
            coverage_ci_upper=row['coverage_ci_upper'],
            density_mean=row['density_mean'],
            density_std=row['density_std'],
            density_ci_lower=row['density_ci_lower'],
            density_ci_upper=row['density_ci_upper'],
            n_agents_mean=int(global_config.n_agents_total * row['coverage_mean']),
            is_selected=is_selected,
            n_replications=int(row['n_replications'])
        ))
    
    return trajectory_enhanced


def generate_prim_trajectory(
    scenario: ScenarioType,
    n_iterations: int,
    n_replications: int,
    noise_std: float,
    random_state: np.random.RandomState
) -> Tuple[List[PRIMTrajectoryEnhancedSchema], List[PRIMTrajectoryReplicationSchema]]:
    """
    Generate complete PRIM trajectory data: both aggregated and disaggregated.
    
    ENHANCED VERSION: Includes replications for uncertainty quantification.
    All parameters loaded from YAML configuration.
    """
    replications = generate_prim_trajectory_replications(
        scenario, n_iterations, n_replications, noise_std, random_state
    )
    
    aggregated = aggregate_prim_trajectory_replications(replications)
    
    return aggregated, replications