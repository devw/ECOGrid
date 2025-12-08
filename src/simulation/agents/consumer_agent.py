from src.simulation.agents.base_agent import BaseAgent
import random


class ConsumerAgent(BaseAgent):
    """Consumer Agent with income in range 0-20,000."""
    
    def __init__(self, unique_id, model, scenario, config):
        super().__init__(unique_id, model, scenario, config)
        
        # Set income based on consumer range from config
        income_config = config.get('income', {}).get('consumer', {})
        income_min = income_config.get('min', 0)
        income_max = income_config.get('max', 20000)
        self.income = random.randint(income_min, income_max)