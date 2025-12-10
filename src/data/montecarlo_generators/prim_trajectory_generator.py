"""
PRIM Trajectory Generation for Agent-Based Model.

This module generates PRIM peeling trajectories with Monte Carlo replications
for uncertainty quantification in energy transition scenarios.
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


def generate_prim_trajectory_replications(
    scenario: ScenarioType,
    n_iterations: int,
    n_replications: int,
    noise_std: float,
    random_state: np.random.RandomState
) -> List[PRIMTrajectoryReplicationSchema]:
    """
    Generate PRIM peeling trajectory data with replications.
    
    Creates multiple runs to capture uncertainty in the peeling process.
    
    Args:
        scenario: Policy scenario
        n_iterations: Number of peeling iterations
        n_replications: Number of Monte Carlo replications
        noise_std: Standard deviation of noise to add
        random_state: Random state for reproducibility
        
    Returns:
        List of trajectory points for all replications
    """
    all_replications = []
    
    # Base trajectory parameters per scenario
    if scenario == ScenarioType.SERVICES_INCENTIVE:
        # SI: Dramatic peeling (100% → 6% coverage, maintaining high density)
        base_coverages = np.linspace(1.0, 0.06, n_iterations)
        base_densities = 0.30 + 0.51 * (1 - base_coverages) ** 0.8
        selected_iteration = n_iterations - 3
        
    elif scenario == ScenarioType.ECONOMIC_INCENTIVE:
        # EI: Moderate peeling (100% → 31% coverage)
        base_coverages = np.linspace(1.0, 0.31, n_iterations)
        base_densities = 0.25 + 0.40 * (1 - base_coverages) ** 0.6
        selected_iteration = n_iterations - 5
        
    else:  # NO_INCENTIVE
        # NI: Flat trajectory (no meaningful peeling)
        base_coverages = np.linspace(1.0, 0.95, n_iterations)
        base_densities = np.full(n_iterations, 0.20)
        selected_iteration = 0
    
    # Generate replications with noise
    for rep_id in range(n_replications):
        for i in range(n_iterations):
            # Add stochastic noise to base trajectory
            coverage_noise = random_state.normal(0, noise_std * 0.01)
            density_noise = random_state.normal(0, noise_std)
            
            coverage = np.clip(base_coverages[i] + coverage_noise, 0.0, 1.0)
            density = np.clip(base_densities[i] + density_noise, 0.0, 1.0)
            
            all_replications.append(PRIMTrajectoryReplicationSchema(
                scenario=scenario,
                iteration=i,
                replication_id=rep_id,
                coverage=coverage,
                density=density,
                n_agents=int(10000 * coverage),
                is_selected=(i == selected_iteration)
            ))
    
    return all_replications


def aggregate_prim_trajectory_replications(
    replications: List[PRIMTrajectoryReplicationSchema]
) -> List[PRIMTrajectoryEnhancedSchema]:
    """
    Aggregate PRIM trajectory replications into summary statistics.
    
    Uses stats_utils to compute confidence intervals.
    
    Args:
        replications: List of all replication data
        
    Returns:
        List of aggregated trajectory points with statistics
    """
    # Convert to DataFrame for easier aggregation
    df = pd.DataFrame([asdict(r) for r in replications])
    
    # Rename replication_id to run_id for compatibility with stats_utils
    df = df.rename(columns={'replication_id': 'run_id'})
    
    # Aggregate using our utility function
    aggregated = aggregate_replications(
        df,
        group_cols=['scenario', 'iteration'],
        value_cols=['coverage', 'density'],
        confidence=0.95,
        ci_method='parametric'
    )
    
    # Convert back to schema objects
    trajectory_enhanced = []
    for _, row in aggregated.iterrows():
        # Get the is_selected flag (same for all replications in group)
        # Use replication_id here since we're working with original data
        original_df = pd.DataFrame([asdict(r) for r in replications])
        is_selected = original_df[
            (original_df['scenario'] == row['scenario']) & 
            (original_df['iteration'] == row['iteration'])
        ]['is_selected'].iloc[0]
        
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
            n_agents_mean=int(10000 * row['coverage_mean']),
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
    
    Args:
        scenario: Policy scenario
        n_iterations: Number of peeling iterations
        n_replications: Number of Monte Carlo replications
        noise_std: Standard deviation of noise
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (aggregated_trajectory, all_replications)
    """
    # Generate all replications
    replications = generate_prim_trajectory_replications(
        scenario, n_iterations, n_replications, noise_std, random_state
    )
    
    # Aggregate with statistics
    aggregated = aggregate_prim_trajectory_replications(replications)
    
    return aggregated, replications