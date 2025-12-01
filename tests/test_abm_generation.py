import pytest
from pathlib import Path
import json
import csv


def test_abm_generation_minimal(tmp_path):
    """
    Test to check that ABM generation runs without errors
    and produces expected output files.
    """
    from src.simulation.model import SimulationModel
    
    # Setup
    n_consumers = 1
    n_prosumers = 1
    n_grid_agents = 1
    n_steps = 2
    seed = 42
    
    # Create output directory
    run_dir = tmp_path / f"run_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = SimulationModel(
        n_consumers=n_consumers,
        n_prosumers=n_prosumers,
        n_grid_agents=n_grid_agents,
        seed=seed
    )
    
    # Verify agents were created
    assert len(list(model.agents)) == n_consumers + n_prosumers + n_grid_agents
    
    # Run simulation steps
    all_steps_data = []
    for step in range(n_steps):
        model.step()
        step_data = [
            {"agent_id": agent.unique_id, "type": type(agent).__name__}
            for agent in model.agents
        ]
        all_steps_data.append({"step": step + 1, "agents": step_data})
        
        # Save CSV per step
        csv_file = run_dir / f"agents_step_{step + 1}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["agent_id", "type"])
            writer.writeheader()
            for agent in step_data:
                writer.writerow(agent)
    
    # Save aggregated JSON
    json_file = run_dir / "simulation_output.json"
    with open(json_file, "w") as f:
        json.dump(all_steps_data, f, indent=2)
    
    # Assertions
    assert run_dir.exists() and run_dir.is_dir()
    
    # Check CSV files were created
    csv_files = list(run_dir.glob("agents_step_*.csv"))
    assert len(csv_files) == n_steps, f"Expected {n_steps} CSV files, found {len(csv_files)}"
    
    # Verify CSV content
    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == n_consumers + n_prosumers + n_grid_agents
            assert all("agent_id" in row and "type" in row for row in rows)
    
    # Check JSON file exists and has correct structure
    assert json_file.exists()
    with open(json_file, "r") as f:
        data = json.load(f)
        assert len(data) == n_steps
        assert all("step" in entry and "agents" in entry for entry in data)
        assert data[0]["step"] == 1
        assert len(data[0]["agents"]) == n_consumers + n_prosumers + n_grid_agents


def test_abm_agent_types(tmp_path):
    """
    Test that correct agent types are created.
    """
    from src.simulation.model import SimulationModel
    from src.simulation.agents.consumer_agent import ConsumerAgent
    from src.simulation.agents.prosumer_agent import ProsumerAgent
    from src.simulation.agents.grid_agent import GridAgent
    
    model = SimulationModel(n_consumers=2, n_prosumers=3, n_grid_agents=1, seed=42)
    
    agents_list = list(model.agents)
    assert len(agents_list) == 6
    
    # Count agent types
    consumer_count = sum(1 for a in agents_list if isinstance(a, ConsumerAgent))
    prosumer_count = sum(1 for a in agents_list if isinstance(a, ProsumerAgent))
    grid_count = sum(1 for a in agents_list if isinstance(a, GridAgent))
    
    assert consumer_count == 2
    assert prosumer_count == 3
    assert grid_count == 1


def test_abm_seed_reproducibility():
    """
    Test that same seed produces same agent IDs.
    """
    from src.simulation.model import SimulationModel
    
    model1 = SimulationModel(n_consumers=3, n_prosumers=2, n_grid_agents=1, seed=123)
    model2 = SimulationModel(n_consumers=3, n_prosumers=2, n_grid_agents=1, seed=123)
    
    agents1 = sorted([a.unique_id for a in model1.agents])
    agents2 = sorted([a.unique_id for a in model2.agents])
    
    assert agents1 == agents2


def test_abm_model_step():
    """
    Test that model.step() executes without errors.
    """
    from src.simulation.model import SimulationModel
    
    model = SimulationModel(n_consumers=1, n_prosumers=1, n_grid_agents=1, seed=42)
    
    # Should not raise any exceptions
    model.step()
    model.step()
    model.step()