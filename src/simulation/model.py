from mesa import Model
from mesa.time import RandomActivation
from .agents.consumer_agent import ConsumerAgent
from .agents.grid_agent import GridAgent
from .agents.prosumer_agent import ProsumerAgent

class SimulationModel(Model):
    """Skeleton MESA model with mock agents."""
    def __init__(self, N_consumers=2, N_prosumers=2, N_grids=1):
        self.schedule = RandomActivation(self)
        # Add mock agents
        for i in range(N_consumers):
            self.schedule.add(ConsumerAgent(i, self))
        for i in range(N_prosumers):
            self.schedule.add(ProsumerAgent(i + N_consumers, self))
        for i in range(N_grids):
            self.schedule.add(GridAgent(i + N_consumers + N_prosumers, self))

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
