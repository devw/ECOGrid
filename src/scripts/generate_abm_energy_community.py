#!/usr/bin/env python3
"""
📦 Generate ABM Energy Community Data

This script initializes an Agent-Based Model (ABM) of an energy community
with mock agents, runs the simulation for a specified number of steps,
and saves outputs (CSV/JSON) for analysis.

It follows the same pattern as MonteCarlo and Dummy data generators.
"""

import argparse
from pathlib import Path
import csv
import json
from simulation.model import SimulationModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate data from ABM simulation of an energy community."
    )
    parser.add_argument("--n-consumers", type=int, default=2, help="Number of Consumer agents")
    parser.add_argument("--n-prosumers", type=int, default=2, help="Number of Prosumer agents")
    parser.add_argument("--n-grids", type=int, default=1, help="Number of Grid agents")
    parser.add_argument("--n-steps", type=int, default=3, help="Number of simulation steps")
    parser.add_argument("--output", type=str, default="data/abm", help="Folder to save simulation outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def save_agents_csv(agents, file_path):
    """Save agent states to CSV"""
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

    # Initialize the model
    model = SimulationModel(
        N_consumers=args.n_consumers,
        N_prosumers=args.n_prosumers,
        N_grids=args.n_grids,
        seed=args.seed
    )

    # Run simulation
    all_steps_data = []
    for step in range(args.n_steps):
        model.step()
        step_data = [
            {"agent_id": agent.unique_id, "type": type(agent).__name__}
            for agent in model.schedule.agents
        ]
        all_steps_data.append({"step": step + 1, "agents": step_data})

        # Save CSV per step (optional, can be aggregated later)
        save_agents_csv(
            step_data,
            run_dir / f"agents_step_{step + 1}.csv"
        )

    # Save full simulation as JSON
    with open(run_dir / "simulation_output.json", "w") as f:
        json.dump(all_steps_data, f, indent=2)

    print(f"✅ ABM simulation ran for {args.n_steps} steps with mock agents.")
    print(f"📂 Outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
