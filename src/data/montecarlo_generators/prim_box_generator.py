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


def calculate_adoption_in_box(
    agents: List[AgentSchema],
    trust_min: float,
    trust_max: float,
    income_min: float,
    income_max: float,
    adoption_func: Callable[[float, float, float], float]
) -> Tuple[float, float, float, float]:
    """
    Calculate adoption statistics inside and outside the PRIM box.
    
    Args:
        agents: List of agents
        trust_min, trust_max: Trust boundaries
        income_min, income_max: Income boundaries
        adoption_func: Function to calculate adoption probability
        
    Returns:
        Tuple of (adoption_in_box, adoption_outside_box, coverage, n_in_box)
    """
    adoptions_in_box = []
    adoptions_outside_box = []
    
    for agent in agents:
        # Calculate adoption probability for this agent
        adoption_prob = adoption_func(agent.trust, agent.income, agent.environmental_concern)
        
        # Check if agent is in the box
        in_box = (trust_min <= agent.trust <= trust_max and 
                  income_min <= agent.income <= income_max)
        
        if in_box:
            adoptions_in_box.append(adoption_prob)
        else:
            adoptions_outside_box.append(adoption_prob)
    
    # Calculate statistics
    avg_adoption_in_box = np.mean(adoptions_in_box) if adoptions_in_box else 0.0
    avg_adoption_outside_box = np.mean(adoptions_outside_box) if adoptions_outside_box else 0.0
    coverage = len(adoptions_in_box) / len(agents) if agents else 0.0
    
    return avg_adoption_in_box, avg_adoption_outside_box, coverage, len(adoptions_in_box)


def calculate_density_and_lift(
    agents: List[AgentSchema],
    trust_min: float,
    trust_max: float,
    income_min: float,
    income_max: float,
    adoption_func: Callable[[float, float, float], float],
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Calculate density (% of high-adoption cases in box) and lift.
    
    Args:
        agents: List of agents
        trust_min, trust_max: Trust boundaries
        income_min, income_max: Income boundaries
        adoption_func: Function to calculate adoption probability
        threshold: Threshold for "high adoption" (default 0.5 = 50%)
        
    Returns:
        Tuple of (density, lift)
    """
    high_adoption_in_box = 0
    total_high_adoption = 0
    total_adoption_in_box = []
    total_adoption_all = []
    
    for agent in agents:
        adoption_prob = adoption_func(agent.trust, agent.income, agent.environmental_concern)
        total_adoption_all.append(adoption_prob)
        
        # Check if this is a "high adoption" case
        is_high_adoption = adoption_prob >= threshold
        if is_high_adoption:
            total_high_adoption += 1
        
        # Check if agent is in the box
        in_box = (trust_min <= agent.trust <= trust_max and 
                  income_min <= agent.income <= income_max)
        
        if in_box:
            total_adoption_in_box.append(adoption_prob)
            if is_high_adoption:
                high_adoption_in_box += 1
    
    # Calculate density: % of high-adoption cases that are in the box
    density = high_adoption_in_box / total_high_adoption if total_high_adoption > 0 else 0.0
    
    # Calculate lift: (avg adoption in box) / (avg adoption overall)
    avg_in_box = np.mean(total_adoption_in_box) if total_adoption_in_box else 0.0
    avg_overall = np.mean(total_adoption_all) if total_adoption_all else 1.0
    lift = avg_in_box / avg_overall if avg_overall > 0 else 1.0
    
    return density, lift


def identify_prim_box(
    scenario: ScenarioType,
    agents: List[AgentSchema],
    adoption_func: Callable[[float, float, float], float]
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Identify PRIM box boundaries for high-adoption segment.
    
    Args:
        scenario: Policy scenario
        agents: List of agents
        adoption_func: Function to calculate adoption
        
    Returns:
        Tuple of (trust_min, trust_max, income_min, income_max, coverage, density, lift)
    """
    # Define box boundaries (these can be optimized with actual PRIM algorithm)
    if scenario == ScenarioType.SERVICES_INCENTIVE:
        # SI: High trust + high income segment
        trust_min, trust_max = 0.65, 1.0
        income_min, income_max = 70.0, 100.0
        threshold = 0.5  # 50% adoption threshold for "high adoption"
        
    elif scenario == ScenarioType.ECONOMIC_INCENTIVE:
        # EI: High trust + moderate-to-high income
        trust_min, trust_max = 0.55, 1.0
        income_min, income_max = 0.0, 30.0
        threshold = 0.5
        
    else:  # NO_INCENTIVE
        # NI: No clear segment (full population)
        trust_min, trust_max = 0.0, 1.0
        income_min, income_max = 0.0, 100.0
        threshold = 0.3  # Lower threshold for baseline
    
    # Calculate actual statistics from data
    avg_in_box, avg_outside, coverage, n_in_box = calculate_adoption_in_box(
        agents, trust_min, trust_max, income_min, income_max, adoption_func
    )
    
    # Calculate density and lift
    density, lift = calculate_density_and_lift(
        agents, trust_min, trust_max, income_min, income_max, 
        adoption_func, threshold
    )
    
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