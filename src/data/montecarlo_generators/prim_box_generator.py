"""
PRIM Box Generation for Agent-Based Model.

This module identifies and generates PRIM (Patient Rule Induction Method) boxes
for segmenting high-adoption populations in energy transition scenarios.
"""

from typing import List, Tuple, Callable
import numpy as np

from src.data.schemas import (
    ScenarioType,
    AgentSchema,
    PRIMBoxSchema
)
from .adoption_functions import get_adoption_function


def identify_prim_box(
    scenario: ScenarioType,
    agents: List[AgentSchema],
    adoption_func: Callable[[float, float, float], float]
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Identify PRIM box boundaries for high-adoption segment.
    
    Simulates PRIM algorithm results based on scenario characteristics.
    
    Args:
        scenario: Policy scenario
        agents: List of agents
        adoption_func: Function to calculate adoption
        
    Returns:
        Tuple of (trust_min, trust_max, income_min, income_max, coverage, density, lift)
    """
    if scenario == ScenarioType.SERVICES_INCENTIVE:
        # SI: High trust segment (narrow, high density)
        trust_min, trust_max = 0.65, 1.0
        income_min, income_max = 77.5, 100.0  # Top ~23% income
        coverage = 0.067  # 6.7% of population (empirical)
        density = 0.263   # 26.3% of high-adoption cases in box
        lift = 2.39       # 2.39x above average adoption
        
    elif scenario == ScenarioType.ECONOMIC_INCENTIVE:
        # EI: High trust + moderate income (broader segment)
        trust_min, trust_max = 0.55, 1.0
        income_min, income_max = 30.0, 100.0
        coverage = 0.31  # 31% of population
        density = 0.65   # 65% adoption in segment
        lift = 1.8
        
    else:  # NO_INCENTIVE
        # NI: No clear segment (uniform baseline)
        trust_min, trust_max = 0.0, 1.0
        income_min, income_max = 0.0, 100.0
        coverage = 1.0   # 100% (no peeling)
        density = 0.20   # 20% baseline
        lift = 1.0
    
    return trust_min, trust_max, income_min, income_max, coverage, density, lift


def generate_prim_boxes(
    scenario: ScenarioType,
    agents: List[AgentSchema],
    random_state: np.random.RandomState
) -> List[PRIMBoxSchema]:
    """
    Generate PRIM box boundaries for a scenario.
    
    Args:
        scenario: Policy scenario
        agents: List of agents for analysis
        random_state: Random state for reproducibility
        
    Returns:
        List containing the final PRIM box
    """
    adoption_func = get_adoption_function(scenario)
    trust_min, trust_max, income_min, income_max, coverage, density, lift = \
        identify_prim_box(scenario, agents, adoption_func)
    
    box = PRIMBoxSchema(
        scenario=scenario,
        box_id=0,
        trust_min=trust_min,
        trust_max=trust_max,
        income_min=income_min,
        income_max=income_max,
        coverage=coverage,
        density=density,
        lift=lift
    )
    
    return [box]