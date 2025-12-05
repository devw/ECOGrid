#!/usr/bin/env python3
"""
📦 Generate ABM Energy Community Data (Mock Version)

This script runs a minimal ABM simulation using mock agents.
It generates CSV/JSON outputs to test imports, script execution,
and pipeline, without implementing real agent behavior.
"""

from pathlib import Path
import csv
import json

# Import unified CLI parser
from src.utils.cli_parser import base_parser

# Import mock model and agents
from src.simulation.model import SimulationModel


def save_agents_csv(agents_data, file_path):
    """Save agent states to CSV (mock data)"""
    fieldnames = ["agent_id", "type"]
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for agent in agents_data:  # agent is already a dict
            writer.writerow(agent)


def main():
    # Define ABM-specific defaults
    defaults = {
        "n_consumers": 2,
        "n_prosumers": 2,
        "n_grid_agents": 1,
        "n_steps": 3,
        "seed": 42,
        "output": "data/abm"
    }

    # Parse CLI arguments using unified parser
    args = base_parser(defaults=defaults).parse_args()

    # Create output folder
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run_seed_{args.seed}"
    run_dir.mkdir(exist_ok=True)

    # Initialize mock model
    model = SimulationModel(
        n_consumers=args.n_consumers,
        n_prosumers=args.n_prosumers,
        n_grid_agents=args.n_grid_agents,
        seed=args.seed
    )

    # Run simulation steps (mock)
    all_steps_data = []
    for step in range(args.n_steps):
        model.step()  # does nothing in mock
        step_data = [
            {"agent_id": agent.unique_id, "type": type(agent).__name__}
            for agent in model.agents
        ]
        all_steps_data.append({"step": step + 1, "agents": step_data})

        # Save CSV per step
        save_agents_csv(
            step_data,
            run_dir / f"agents_step_{step + 1}.csv"
        )

    # Save aggregated JSON
    with open(run_dir / "simulation_output.json", "w") as f:
        json.dump(all_steps_data, f, indent=2)

    print(f"✅ Mock ABM simulation ran for {args.n_steps} steps.")
    print(f"📂 Outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
