#!/usr/bin/env python3
"""
CLI script to generate dummy data for ECOGrid simulation.
"""

from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.cli_parser import base_parser, safe_run
from src.data.montecarlo_generators.dummy_generator import (
    GeneratorConfig,
    generate_all_scenarios,
    save_all_data
)


if __name__ == "__main__":
    def run():
        # Define defaults for Montecarlo + ABM
        defaults = {
            # Montecarlo
            "n_agents": 10000,
            "n_bins": 20,
            "n_replications": 100,
            "noise_std": 0.05,
            # ABM
            "n_consumers": 2,
            "n_prosumers": 2,
            "n_grid_agents": 1,
            "n_steps": 3,
            # Common
            "seed": 42,
            "output": Path("data/dummy")
        }

        # Parse arguments using unified CLI parser
        args = base_parser(defaults=defaults).parse_args()

        # Create configuration
        config = GeneratorConfig(
            n_agents=args.n_agents,
            n_bins=args.n_bins,
            n_replications=args.n_replications,
            noise_std=args.noise_std,
            random_seed=args.seed,
            output_dir=args.output
        )

        # Display configuration
        print("=" * 70)
        print("🚀 ECOGrid Dummy Data Generator (Enhanced v2.0)")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  • Agents per scenario: {config.n_agents:,}")
        print(f"  • Heatmap bins: {config.n_bins}×{config.n_bins}")
        print(f"  • Monte Carlo replications: {config.n_replications}")
        print(f"  • Noise std: {config.noise_std}")
        print(f"  • Random seed: {config.random_seed}")
        print(f"  • Output directory: {config.output_dir}")

        # Generate and save data
        all_data = generate_all_scenarios(config)
        save_all_data(all_data, config.output_dir, config)

        print("\n" + "=" * 70)
        print("✅ Data generation complete!")
        print("=" * 70)
        print("\n📊 Generated files:")
        print(f"  • {config.output_dir / 'heatmap_grid.csv'} (with std_dev, CI)")
        print(f"  • {config.output_dir / 'heatmap_replications.csv'} (all {config.n_replications} runs)")
        print(f"  • {config.output_dir / 'prim_boxes.csv'}")
        print(f"  • {config.output_dir / 'prim_trajectory.csv'}")
        print(f"  • {config.output_dir / 'demographic_profiles.csv'}")
        print(f"  • {config.output_dir / 'scale_metadata.json'} (scale documentation)")
        print("\n📈 Ready for enhanced visualization:")
        print("  • Figure 1: Statistical heatmaps with confidence intervals")
        print("  • Statistical significance testing between scenarios")
        print("  • Full uncertainty quantification")

    # Run safely with standardized error handling
    safe_run(run)
