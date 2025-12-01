# src/simulation/model.py
from simulation.agents.base_agent import BaseAgent
from simulation.schedulers import MockScheduler

class SimulationModel:
    def __init__(self, N_consumers=2, N_prosumers=2, N_grids=1, seed=42):
        self.schedule = MockScheduler()
        # create mock agents
        for i in range(N_consumers):
            self.schedule.add(BaseAgent(unique_id=f"C{i+1}"))
        for i in range(N_prosumers):
            self.schedule.add(BaseAgent(unique_id=f"P{i+1}"))
        for i in range(N_grids):
            self.schedule.add(BaseAgent(unique_id=f"G{i+1}"))

    def step(self):
        # mock step: do nothing
        pass
