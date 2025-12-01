#!/usr/bin/env python3
"""
📦 Generate ABM Energy Community Data (Mock Version)

This script runs a minimal ABM simulation using mock agents.
It generates CSV/JSON outputs to test imports, script execution,
and pipeline, without implementing real agent behavior.
"""

import argparse
from pathlib import Path
import csv
import json

# Import mock model and agents
from simulation.model import SimulationModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mock ABM data for Energy Community."
    )
    parser.add_argument("--n-consumers", type=int, default=2, help="Number of Consumer agents")
    parser.add_argument("--n-prosumers", type=int, default=2, help="Number of Prosumer agents")
    parser.add_argument("--n-grids", type=int, default=1, help="Number of Grid agents")
    parser.add_argument("--n-steps", type=int, default=3, help="Number of simulation steps")
    parser.add_argument("--output", type=str, default="data/abm", help="Folder to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def save_agents_csv(agents, file_path):
    """Save agent states to CSV (mock data)"""
    fieldnames = ["agent_id", "type"]
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for agent in agents:
            writer.writerow({"agent_id": agent.unique_id, "type": type(agent).__name__})


def main():
    args = parse_args()

    # Create output folder
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run_seed_{args.seed}"
    run_dir.mkdir(exist_ok=True)

    # Initialize mock model
    model = SimulationModel(
        N_consumers=args.n_consumers,
        N_prosumers=args.n_prosumers,
        N_grids=args.n_grids,
        seed=args.seed
    )

    # Run simulation steps (mock)
    all_steps_data = []
    for step in range(args.n_steps):
        model.step()  # does nothing in mock
        step_data = [
            {"agent_id": agent.unique_id, "type": type(agent).__name__}
            for agent in model.schedule.agents
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
