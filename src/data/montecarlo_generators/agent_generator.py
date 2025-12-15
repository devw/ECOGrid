"""
Agent Generation for Agent-Based Model.

This module generates individual agent data with trust and income attributes
for energy transition scenario simulations.
"""

from typing import List
import numpy as np

from src.data.schemas import ScenarioType, AgentSchema
from src.data.montecarlo_generators.adoption_functions import get_adoption_function


def generate_agents(
    scenario: ScenarioType,
    n_agents: int,
    random_state: np.random.RandomState
) -> List[AgentSchema]:
    agents = []
    adoption_func = get_adoption_function(scenario)  # ← AGGIUNTO
    
    for agent_id in range(n_agents):
        trust = random_state.beta(2, 2)
        income = random_state.lognormal(mean=3.5, sigma=0.85)
        income = np.clip(income, 0.0, 100.0)
        environmental_concern = random_state.beta(2.5, 2)
        
        adoption_rate = adoption_func(trust, income) 
        
        agents.append(AgentSchema(
            agent_id=agent_id,
            trust=trust,
            income=income,
            environmental_concern=environmental_concern,
            scenario=scenario,
            adoption_rate=adoption_rate  # ← AGGIUNTO
        ))
    
    return agents