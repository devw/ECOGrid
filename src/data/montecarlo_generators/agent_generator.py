"""
Agent Generation for Agent-Based Model - FIXED VERSION.

Changes:
1. Added individual agent noise (config.noise_std)
2. Noise applied BEFORE clipping to [0, 1]
3. This increases variability → reduces Cohen's d to realistic levels (0.8-1.2)
"""

from typing import List
import numpy as np

from src.data.schemas import ScenarioType, AgentSchema
from src.data.montecarlo_generators.adoption_functions import get_adoption_function


def generate_agents(
    scenario: ScenarioType,
    n_agents: int,
    random_state: np.random.RandomState,
    noise_std: float = 0.12
) -> List[AgentSchema]:
    """
    Generate agents with realistic adoption variability.
    
    Args:
        scenario: Policy scenario type
        n_agents: Number of agents to generate
        random_state: NumPy random state for reproducibility
        noise_std: Standard deviation of individual agent noise
        
    Returns:
        List of agent schemas with adoption rates
    """
    agents = []
    adoption_func = get_adoption_function(scenario)
    
    for agent_id in range(n_agents):
        # Generate agent characteristics
        trust = random_state.beta(2, 2)
        income = random_state.lognormal(mean=3.5, sigma=0.85)
        income = np.clip(income, 0.0, 100.0)
        environmental_concern = random_state.beta(2.5, 2)
        
        # Calculate base adoption rate from model
        adoption_rate = adoption_func(trust, income)
        
        # Add individual agent noise to increase variability
        noise = random_state.normal(0, noise_std)
        adoption_rate = adoption_rate + noise
        
        # Clip to valid probability range
        adoption_rate = np.clip(adoption_rate, 0.0, 1.0)
        
        agents.append(AgentSchema(
            agent_id=agent_id,
            trust=trust,
            income=income,
            environmental_concern=environmental_concern,
            scenario=scenario,
            adoption_rate=adoption_rate
        ))
    
    return agents