"""
Demographic Profile Generation for Agent-Based Model.

This module generates demographic profile analysis data (Table III)
for identified PRIM segments in energy transition scenarios.
"""

from typing import List

from src.data.schemas import (
    ScenarioType,
    AgentSchema,
    PRIMBoxSchema,
    DemographicProfileSchema
)


def generate_demographic_profiles(
    scenario: ScenarioType,
    agents: List[AgentSchema],
    prim_boxes: List[PRIMBoxSchema]
) -> List[DemographicProfileSchema]:
    """
    Generate demographic profile analysis data (Table III).
    
    Args:
        scenario: Policy scenario
        agents: List of agents
        prim_boxes: PRIM boxes identified for scenario
        
    Returns:
        List containing demographic profile for the scenario
    """
    if not prim_boxes:
        raise ValueError("No PRIM boxes provided")
    
    box = prim_boxes[0]
    n_agents_total = len(agents)
    n_agents_segment = int(n_agents_total * box.coverage)
    
    # Generate segment name based on scenario characteristics
    if scenario == ScenarioType.SERVICES_INCENTIVE:
        segment_name = "High Trust Community (Trust ≥ 0.65)"
    elif scenario == ScenarioType.ECONOMIC_INCENTIVE:
        segment_name = "High Trust + Mid-High Income (Trust ≥ 0.55, Income ≥ 30)"
    else:
        segment_name = "Baseline Population (No Segmentation)"
    
    profile = DemographicProfileSchema(
        scenario=scenario,
        segment_name=segment_name,
        trust_min=box.trust_min,
        trust_max=box.trust_max,
        income_min=box.income_min,
        income_max=box.income_max,
        coverage=box.coverage,
        density=box.density,
        lift=box.lift,
        n_agents_total=n_agents_total,
        n_agents_segment=n_agents_segment
    )
    
    return [profile]