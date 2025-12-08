#!/usr/bin/env python3
"""
📦 Generate ABM Energy Community Data
This script runs an ABM simulation with configurable agents.
It generates CSV/JSON outputs with agent states including trust,
income, adoption, and grid assignments.
"""
from pathlib import Path
import csv
import json

# Import unified CLI parser
from src.utils.cli_parser import base_parser

# Import simulation model
from src.simulation.model import SimulationModel


def save_agents_csv(agents_data, step, file_path):
    """
    Save agent states to CSV with all required columns.
    
    Args:
        agents_data: List of agent state dictionaries
        step: Current simulation step number
        file_path: Path to save CSV file
    """
    if not agents_data:
        return
    
    # Define fieldnames with step included
    fieldnames = ["scenario", "step", "agent_id", "type", "trust", 
                  "income", "adoption", "grid_id"]
    
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for agent in agents_data:
            # Add step to each row
            row = agent.copy()
            row["step"] = step
            writer.writerow(row)


def main():
    # Define ABM-specific defaults
    defaults = {
        "n_consumers": 2,
        "n_prosumers": 2,
        "n_grid_agents": 1,
        "n_steps": 3,
        "seed": 42,
        "output": "data/abm",
        "scenario": "default",
        "config": "config/abm_config.yaml"
    }
    
    # Parse CLI arguments using unified parser
    parser = base_parser(defaults=defaults)
    
    # Add ABM-specific arguments
    parser.add_argument(
        "--scenario",
        type=str,
        default=defaults["scenario"],
        help="Scenario name for this simulation run (e.g., baseline, high-trust, etc.)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=defaults["config"],
        help="Path to YAML configuration file for simulation parameters"
    )
    
    args = parser.parse_args()
    
    # Create output folder
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run_seed_{args.seed}"
    run_dir.mkdir(exist_ok=True)
    
    # Initialize simulation model with config
    print(f"🎯 Initializing simulation: scenario='{args.scenario}'")
    print(f"📋 Config: {args.config}")
    print(f"👥 Agents: {args.n_consumers} consumers, {args.n_prosumers} prosumers, {args.n_grid_agents} grid")
    
    model = SimulationModel(
        n_consumers=args.n_consumers,
        n_prosumers=args.n_prosumers,
        n_grid_agents=args.n_grid_agents,
        scenario=args.scenario,
        config_path=args.config,
        seed=args.seed
    )
    
    # Run simulation steps
    all_steps_data = []
    
    for step in range(1, args.n_steps + 1):
        # Step the model (updates trust and other dynamic properties)
        model.step()
        
        # Get current agent states
        agent_states = model.get_all_agent_states()
        
        # Store for JSON output
        all_steps_data.append({
            "step": step,
            "scenario": args.scenario,
            "agents": agent_states
        })
        
        # Save CSV per step
        save_agents_csv(
            agent_states,
            step,
            run_dir / f"agents_step_{step}.csv"
        )
        
        print(f"  ✓ Step {step}/{args.n_steps} completed")
    
    # Save aggregated JSON
    output_json = {
        "scenario": args.scenario,
        "seed": args.seed,
        "n_consumers": args.n_consumers,
        "n_prosumers": args.n_prosumers,
        "n_grid_agents": args.n_grid_agents,
        "n_steps": args.n_steps,
        "config_path": args.config,
        "steps": all_steps_data
    }
    
    with open(run_dir / "simulation_output.json", "w") as f:
        json.dump(output_json, f, indent=2)
    
    print(f"\n✅ ABM simulation completed!")
    print(f"📊 Scenario: {args.scenario}")
    print(f"🔢 Steps: {args.n_steps}")
    print(f"📂 Outputs saved to: {run_dir}")
    print(f"   - CSV files: agents_step_*.csv")
    print(f"   - JSON file: simulation_output.json")


if __name__ == "__main__":
    main()