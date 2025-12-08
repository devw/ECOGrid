from src.simulation.agents.base_agent import BaseAgent
import random


class ProsumerAgent(BaseAgent):
    """Prosumer Agent with income in range 20,000-50,000."""
    
    def __init__(self, unique_id, model, scenario, config):
        super().__init__(unique_id, model, scenario, config)
        
        # Set income based on prosumer range from config
        income_config = config.get('income', {}).get('prosumer', {})
        income_min = income_config.get('min', 20000)
        income_max = income_config.get('max', 50000)
        self.income = random.randint(income_min, income_max)