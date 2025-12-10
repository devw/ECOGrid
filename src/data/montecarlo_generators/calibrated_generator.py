"""
Dummy data generator for ABM energy transition simulation.

This module generates realistic synthetic data for visualization and analysis
development, following the patterns described in the research paper.

ENHANCED VERSION: Includes statistical significance metrics, confidence intervals,
and replications for robust uncertainty quantification.

Principles:
- Functional approach: pure functions with explicit inputs/outputs
- SOLID: Single responsibility per function
- DRY: Reusable utilities for common operations
"""

from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import sys
import json

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.data.schemas import ScenarioType
from .config import GeneratorConfig
from src.data.csv_utils import schemas_to_csv

# =============================================================================
# NEW: Enhanced Schemas for PRIM Trajectory with Replications
# =============================================================================

@dataclass
class PRIMTrajectoryReplicationSchema:
    """Individual replication data for PRIM trajectory uncertainty analysis."""
    scenario: ScenarioType
    iteration: int
    replication_id: int
    coverage: float
    density: float
    n_agents: int
    is_selected: bool


@dataclass
class PRIMTrajectoryEnhancedSchema:
    """Enhanced PRIM trajectory with statistical metrics."""
    scenario: ScenarioType
    iteration: int
    coverage_mean: float
    coverage_std: float
    coverage_ci_lower: float
    coverage_ci_upper: float
    density_mean: float
    density_std: float
    density_ci_lower: float
    density_ci_upper: float
    n_agents_mean: float
    is_selected: bool
    n_replications: int

@dataclass
class HeatmapReplicationSchema:
    """Individual replication data for uncertainty analysis."""
    scenario: ScenarioType
    trust_bin: float
    income_bin: float
    replication_id: int
    adoption_rate: float
    n_samples: int

from .metadata_generator import generate_scale_metadata
from .orchestrator import generate_all_scenarios
from .csv_writer import save_all_data

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function for standalone usage."""
    config = GeneratorConfig()
    
    print("=" * 70)
    print("🚀 ECOGrid Dummy Data Generator (Enhanced v2.0)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  • Agents per scenario: {config.n_agents:,}")
    print(f"  • Heatmap bins: {config.n_bins}×{config.n_bins}")
    print(f"  • Monte Carlo replications: {config.n_replications}")
    print(f"  • Noise std dev: {config.noise_std}")
    print(f"  • Random seed: {config.random_seed}")
    print(f"  • Output directory: {config.output_dir}")
    
    # Generate all scenario data
    all_data = generate_all_scenarios(config)
    
    # Save to CSV files
    save_all_data(all_data, config.output_dir, config)
    
    print("\n" + "=" * 70)
    print("✅ Data generation complete!")
    print("=" * 70)
    print("\n📊 Generated files:")
    print("  • heatmap_grid.csv (with std_dev, CI)")
    print("  • heatmap_replications.csv (all 100 runs)")
    print("  • prim_boxes.csv")
    print("  • prim_trajectory.csv")
    print("  • demographic_profiles.csv")
    print("  • scale_metadata.json (NEW)")
    print("\n📈 Ready for enhanced visualization with:")
    print("  • Statistical significance testing")
    print("  • Confidence interval error bars")
    print("  • Full uncertainty quantification")

if __name__ == "__main__":
    main()