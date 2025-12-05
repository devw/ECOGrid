# src/simulation/agents/base_agent.py
from mesa import Agent

class BaseAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(model)  # Mesa 3.x only takes 'model' argument
        self.unique_id = unique_id  # Set unique_id manually
        self.mock_attribute = 0