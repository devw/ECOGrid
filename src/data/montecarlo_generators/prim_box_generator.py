"""
PRIM Box Generation - Unified Config.
Uses scenario_config for all parameters.
"""
from typing import List, Tuple, Callable
import numpy as np
from src.data.schemas import ScenarioType, AgentSchema, PRIMBoxSchema
from .adoption_functions import get_adoption_function
from .scenario_config import get_prim_box_config


def _partition_agents_by_box(
    agents: List[AgentSchema],
    trust_min: float,
    trust_max: float,
    income_min: float,
    income_max: float,
    adoption_func: Callable[[float, float], float]
) -> Tuple[List[float], List[float]]:
    """Single-pass agent partitioning."""
    adoptions_in = []
    adoptions_out = []
    
    for agent in agents:
        adoption = adoption_func(agent.trust, agent.income)
        in_box = (trust_min <= agent.trust <= trust_max and 
                  income_min <= agent.income <= income_max)
        (adoptions_in if in_box else adoptions_out).append(adoption)
    
    return adoptions_in, adoptions_out


def _calculate_statistics(
    adoptions_in: List[float],
    adoptions_out: List[float],
    threshold: float,
    n_total: int
) -> Tuple[float, float, float, float, float]:
    """Calculate PRIM statistics."""
    n_in = len(adoptions_in)
    all_adoptions = adoptions_in + adoptions_out
    
    avg_in = np.mean(adoptions_in) if adoptions_in else 0.0
    avg_out = np.mean(adoptions_out) if adoptions_out else 0.0
    coverage = n_in / n_total if n_total > 0 else 0.0
    
    high_in = sum(1 for a in adoptions_in if a >= threshold)
    density = high_in / n_in if n_in > 0 else 0.0
    
    avg_all = np.mean(all_adoptions) if all_adoptions else 1.0
    lift = avg_in / avg_all if avg_all > 0 else 1.0
    
    return avg_in, avg_out, coverage, density, lift


def identify_prim_box(
    scenario: ScenarioType,
    agents: List[AgentSchema],
    adoption_func: Callable[[float, float], float]
) -> Tuple[float, float, float, float, float, float, float]:
    """Identify PRIM box from unified config."""
    config = get_prim_box_config(scenario)
    
    adoptions_in, adoptions_out = _partition_agents_by_box(
        agents,
        config.trust_min,
        config.trust_max,
        config.income_min,
        config.income_max,
        adoption_func
    )
    
    _, _, coverage, density, lift = _calculate_statistics(
        adoptions_in,
        adoptions_out,
        config.threshold,
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
    """Generate PRIM boxes for scenario."""
    adoption_func = get_adoption_function(scenario)
    trust_min, trust_max, income_min, income_max, coverage, density, lift = \
        identify_prim_box(scenario, agents, adoption_func)
    
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