"""
Agent Generation for Agent-Based Model.

This module generates individual agent data with trust and income attributes
for energy transition scenario simulations.
"""

from typing import List
import numpy as np

from src.data.schemas import ScenarioType, AgentSchema


def generate_agents(
    scenario: ScenarioType,
    n_agents: int,
    random_state: np.random.RandomState
) -> List[AgentSchema]:
    """
    Generate individual agent data with trust and income attributes.
    
    Args:
        scenario: Policy scenario
        n_agents: Number of agents to generate
        random_state: Random state for reproducibility
        
    Returns:
        List of validated agent schemas
    """
    agents = []
    
    for agent_id in range(n_agents):
        # Generate trust and income from realistic distributions
        trust = random_state.beta(2, 2)  # Beta distribution centered around 0.5
        income = random_state.lognormal(mean=3.5, sigma=0.6)  # Log-normal income
        income = np.clip(income, 0.0, 100.0)
        
        agents.append(AgentSchema(
            agent_id=agent_id,
            trust=trust,
            income=income,
            scenario=scenario
        ))
    
    return agents