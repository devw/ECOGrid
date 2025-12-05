# src/simulation/schedulers.py
class MockScheduler:
    def __init__(self):
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)
