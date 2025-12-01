from mesa import Model
from src.simulation.agents.base_agent import BaseAgent
from src.simulation.agents.consumer_agent import ConsumerAgent
from src.simulation.agents.prosumer_agent import ProsumerAgent
from src.simulation.agents.grid_agent import GridAgent


class SimulationModel(Model):
    """
    Skeleton of the Energy Community ABM using Mesa.
    This model creates mock agents and steps through a few iterations.
    """
    def __init__(self, n_consumers=5, n_prosumers=5, n_grid_agents=1, seed=None):
        super().__init__(seed=seed)
        
        # Create Consumer agents
        for i in range(n_consumers):
            agent = ConsumerAgent(i, self)
            # Agents are automatically registered when created in Mesa 3.x
        
        # Create Prosumer agents
        for i in range(n_consumers, n_consumers + n_prosumers):
            agent = ProsumerAgent(i, self)
        
        # Create Grid agents
        for i in range(n_consumers + n_prosumers, n_consumers + n_prosumers + n_grid_agents):
            agent = GridAgent(i, self)
    
    def step(self):
        """
        Advance the model by one step.
        In Mesa 3.x, we iterate through self.agents directly.
        Agents are shuffled randomly by default when using agents.shuffle().
        """
        # Shuffle agents for random activation order (like RandomActivation)
        for agent in self.agents.shuffle():
            agent.step()