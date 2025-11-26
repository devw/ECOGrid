#!/usr/bin/env python3
"""
CLI script to generate dummy data for ECOGrid simulation.

Usage:
    python src/scripts/generate_dummy_data.py
    python src/scripts/generate_dummy_data.py --n-agents 5000 --n-bins 15
    python src/scripts/generate_dummy_data.py --seed 123 --output data/custom
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.generators.dummy_generator import (
    GeneratorConfig,
    generate_all_scenarios,
    save_all_data
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate dummy data for ABM energy transition simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--n-agents',
        type=int,
        default=10000,
        help='Number of agents per scenario'
    )
    
    parser.add_argument(
        '--n-bins',
        type=int,
        default=20,
        help='Number of bins per dimension for heatmap'
    )
    
    parser.add_argument(
        '--noise-std',
        type=float,
        default=0.05,
        help='Standard deviation of noise in adoption rates'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/dummy'),
        help='Output directory for generated CSV files'
    )
    
    return parser.parse_args()


def main():
    """Main CLI execution."""
    args = parse_arguments()
    
    # Create configuration from CLI arguments
    config = GeneratorConfig(
        n_agents=args.n_agents,
        n_bins=args.n_bins,
        noise_std=args.noise_std,
        random_seed=args.seed,
        output_dir=args.output
    )
    
    print("=" * 70)
    print("🚀 ECOGrid Dummy Data Generator")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  • Agents per scenario: {config.n_agents:,}")
    print(f"  • Heatmap bins: {config.n_bins}×{config.n_bins}")
    print(f"  • Noise std: {config.noise_std}")
    print(f"  • Random seed: {config.random_seed}")
    print(f"  • Output directory: {config.output_dir}")
    
    try:
        # Generate all scenario data
        all_data = generate_all_scenarios(config)
        
        # Save to CSV files
        save_all_data(all_data, config.output_dir)
        
        print("\n" + "=" * 70)
        print("✅ Data generation complete!")
        print("=" * 70)
        print("\n📊 Generated files:")
        print(f"  • {config.output_dir / 'heatmap_grid.csv'}")
        print(f"  • {config.output_dir / 'prim_boxes.csv'}")
        print(f"  • {config.output_dir / 'prim_trajectory.csv'}")
        print(f"  • {config.output_dir / 'demographic_profiles.csv'}")
        print("\n📈 Ready for visualization:")
        print("  • Figure 1 (Heatmaps): Use heatmap_grid.csv + prim_boxes.csv")
        print("  • Figure 2 (Trajectory): Use prim_trajectory.csv")
        print("  • Table III (Demographics): Use demographic_profiles.csv")
        
    except Exception as e:
        print(f"\n❌ Error during data generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()