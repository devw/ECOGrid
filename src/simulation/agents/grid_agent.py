from src.simulation.agents.base_agent import BaseAgent


class GridAgent(BaseAgent):
    """Grid Agent - no income, no grid_id assignment."""
    
    def __init__(self, unique_id, model, scenario, config):
        super().__init__(unique_id, model, scenario, config)
        
        # Grid agents have no income
        self.income = None
        
        # Grid agents don't get assigned to another grid
        self.grid_id = None