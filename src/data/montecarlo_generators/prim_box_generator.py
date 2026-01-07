"""
PRIM Box Generation - FIXED VERSION.

Key fixes:
1. Uses agent.adoption_rate (includes noise) instead of recalculating
2. Density = mean adoption rate (not binary threshold percentage)
3. Consistent with demographic_generator.py calculations
"""
from typing import List, Tuple
import numpy as np
from src.data.schemas import ScenarioType, AgentSchema, PRIMBoxSchema
from .scenario_config import get_prim_box_config


def _partition_agents_by_box(
    agents: List[AgentSchema],
    trust_min: float,
    trust_max: float,
    income_min: float,
    income_max: float
) -> Tuple[List[float], List[float]]:
    """
    Single-pass agent partitioning.
    
    FIX: Uses agent.adoption_rate directly (includes noise and clipping).
    """
    adoptions_in = []
    adoptions_out = []
    
    for agent in agents:
        # Use pre-calculated adoption rate (includes noise)
        adoption = agent.adoption_rate
        
        in_box = (trust_min <= agent.trust <= trust_max and 
                  income_min <= agent.income <= income_max)
        
        (adoptions_in if in_box else adoptions_out).append(adoption)
    
    return adoptions_in, adoptions_out


def _calculate_statistics(
    adoptions_in: List[float],
    adoptions_out: List[float],
    n_total: int
) -> Tuple[float, float, float, float, float]:
    """
    Calculate PRIM statistics.
    
    FIX: Density = mean adoption rate (not threshold-based percentage).
    """
    n_in = len(adoptions_in)
    all_adoptions = adoptions_in + adoptions_out
    
    avg_in = np.mean(adoptions_in) if adoptions_in else 0.0
    avg_out = np.mean(adoptions_out) if adoptions_out else 0.0
    coverage = n_in / n_total if n_total > 0 else 0.0
    
    # FIX: Density is simply the mean adoption rate inside the box
    density = avg_in
    
    # Lift: ratio of in-box adoption to overall adoption
    avg_all = np.mean(all_adoptions) if all_adoptions else 1.0
    lift = avg_in / avg_all if avg_all > 0 else 1.0
    
    return avg_in, avg_out, coverage, density, lift


def identify_prim_box(
    scenario: ScenarioType,
    agents: List[AgentSchema]
) -> Tuple[float, float, float, float, float, float, float]:
    """Identify PRIM box from unified config."""
    config = get_prim_box_config(scenario)
    
    adoptions_in, adoptions_out = _partition_agents_by_box(
        agents,
        config.trust_min,
        config.trust_max,
        config.income_min,
        config.income_max
    )
    
    _, _, coverage, density, lift = _calculate_statistics(
        adoptions_in,
        adoptions_out,
        len(agents)
    )
    
    return (config.trust_min, config.trust_max, 
            config.income_min, config.income_max,
            coverage, density, lift)


def generate_prim_boxes(
    scenario: ScenarioType,
    agents: List[AgentSchema],
    random_state: np.random.RandomState
) -> List[PRIMBoxSchema]:
    """
    Generate PRIM boxes for scenario.
    
    FIX: Removed unused adoption_func parameter and random_state usage.
    """
    trust_min, trust_max, income_min, income_max, coverage, density, lift = \
        identify_prim_box(scenario, agents)
    
    return [PRIMBoxSchema(
        scenario=scenario,
        box_id=0,
        trust_min=trust_min,
        trust_max=trust_max,
        income_min=income_min,
        income_max=income_max,
        coverage=coverage,
        density=density,
        lift=lift
    )]