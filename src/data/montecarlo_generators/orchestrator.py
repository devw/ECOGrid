"""
Orchestration for Monte Carlo Data Generation.

This module coordinates the generation of all data types across policy scenarios
for the agent-based energy transition model.
"""

import numpy as np

from src.data.schemas import ScenarioType
from .config import GeneratorConfig
from .agent_generator import generate_agents
from .heatmap_generator import generate_heatmap_grid
from .prim_box_generator import generate_prim_boxes
from .prim_trajectory_generator import generate_prim_trajectory
from .demographic_generator import generate_demographic_profiles


def generate_scenario_data(
    scenario: ScenarioType,
    config: GeneratorConfig
) -> dict:
    """
    Generate all data types for a single scenario.
    
    Args:
        scenario: Policy scenario to generate
        config: Generator configuration
        
    Returns:
        Dictionary with all generated data types
    """
    random_state = np.random.RandomState(config.random_seed)
    
    print(f"\n🔄 Generating data for scenario: {scenario.value}")
    
    # Generate agents
    print(f"  ├─ Generating {config.n_agents} agents...")
    agents = generate_agents(scenario, config.n_agents, random_state)
    
    # Generate heatmap grid WITH replications
    print(f"  ├─ Generating {config.n_bins}×{config.n_bins} heatmap grid "
          f"({config.n_replications} replications)...")
    heatmap_grid, heatmap_replications = generate_heatmap_grid(
        scenario, config.n_bins, config.n_replications, config.noise_std, random_state
    )
    
    # Generate PRIM analysis
    print(f"  ├─ Identifying PRIM boxes...")
    prim_boxes = generate_prim_boxes(scenario, agents, random_state)
    
    # Generate PRIM trajectory WITH replications
    print(f"  ├─ Generating PRIM trajectory ({config.n_replications} replications)...")
    prim_trajectory_summary, prim_trajectory_replications = generate_prim_trajectory(
        scenario, 15, config.n_replications, config.noise_std, random_state
    )
    
    print(f"  └─ Generating demographic profiles...")
    demographic_profiles = generate_demographic_profiles(scenario, agents, prim_boxes)
    
    return {
        'agents': agents,
        'heatmap_grid': heatmap_grid,
        'heatmap_replications': heatmap_replications,
        'prim_boxes': prim_boxes,
        'prim_trajectory_summary': prim_trajectory_summary,
        'prim_trajectory_replications': prim_trajectory_replications,
        'demographic_profiles': demographic_profiles
    }


def generate_all_scenarios(config: GeneratorConfig) -> dict:
    """
    Generate data for all policy scenarios.
    
    Args:
        config: Generator configuration
        
    Returns:
        Dictionary mapping scenario names to their data
    """
    all_data = {}
    
    scenarios = [
        ScenarioType.NO_INCENTIVE,
        ScenarioType.SERVICES_INCENTIVE,
        ScenarioType.ECONOMIC_INCENTIVE
    ]
    
    for scenario in scenarios:
        all_data[scenario.value] = generate_scenario_data(scenario, config)
    
    return all_data