from mesa import Agent
import random


class BaseAgent(Agent):
    """Base agent with common properties for all agent types."""
    
    def __init__(self, unique_id, model, scenario, config):
        super().__init__(model)
        self.unique_id = unique_id
        self.scenario = scenario
        self.config = config
        
        # Initialize trust (will evolve over time)
        trust_config = config.get('trust', {})
        self.trust = random.uniform(
            trust_config.get('initial_min', 0.5),
            trust_config.get('initial_max', 1.0)
        )
        self.trust_min = trust_config.get('min_value', 0.0)
        self.trust_max = trust_config.get('max_value', 1.0)
        self.trust_change_min = trust_config.get('change_min', -0.05)
        self.trust_change_max = trust_config.get('change_max', 0.05)
        
        # Initialize adoption state (0 = not adopted, 1 = fully adopted)
        adoption_config = config.get('adoption', {})
        self.adoption_state = random.uniform(
            adoption_config.get('initial_min', 0.0),
            adoption_config.get('initial_max', 0.0)
        )
        
        # Income (fixed, will be set by subclasses)
        self.income = None
        
        # Grid assignment (will be set by model)
        self.grid_id = None
    
    def step(self):
        """
        Update agent state each step.
        Currently updates trust with random walk.
        """
        self.update_trust()
    
    def update_trust(self):
        """
        Evolve trust over time with random changes.
        Trust is bounded between trust_min and trust_max.
        """
        change = random.uniform(self.trust_change_min, self.trust_change_max)
        self.trust += change
        
        # Ensure trust stays within bounds
        self.trust = max(self.trust_min, min(self.trust_max, self.trust))
    
    def get_state(self):
        """
        Return current agent state as a dictionary.
        Used for CSV/JSON output.
        """
        return {
            'agent_id': self.unique_id,
            'type': type(self).__name__,
            'trust': round(self.trust, 4),
            'income': self.income,
            'adoption': round(self.adoption_state, 4),
            'grid_id': self.grid_id,
            'scenario': self.scenario
        }