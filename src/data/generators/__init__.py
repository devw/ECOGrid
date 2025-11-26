"""
Data generators package for ECOGrid simulation.

This package contains modules for generating synthetic data for testing,
visualization development, and analysis validation.
"""

from src.data.generators.dummy_generator import (
    GeneratorConfig,
    generate_agents,
    generate_heatmap_grid,
    generate_prim_boxes,
    generate_prim_trajectory,
    generate_demographic_profiles,
    generate_scenario_data,
    generate_all_scenarios,
    save_all_data,
    calculate_adoption_no_incentive,
    calculate_adoption_services_incentive,
    calculate_adoption_economic_incentive,
    get_adoption_function,
)

__all__ = [
    'GeneratorConfig',
    'generate_agents',
    'generate_heatmap_grid',
    'generate_prim_boxes',
    'generate_prim_trajectory',
    'generate_demographic_profiles',
    'generate_scenario_data',
    'generate_all_scenarios',
    'save_all_data',
    'calculate_adoption_no_incentive',
    'calculate_adoption_services_incentive',
    'calculate_adoption_economic_incentive',
    'get_adoption_function',
]