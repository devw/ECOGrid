from simulation.model import SimulationModel

if __name__ == "__main__":
    model = SimulationModel()
    for _ in range(3):  # run 3 steps
        model.step()
    print("Simulation ran successfully with mock agents.")
