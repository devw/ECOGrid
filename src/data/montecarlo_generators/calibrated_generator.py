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

from src.data.schemas import (
    ScenarioType,
    AgentSchema,
    PRIMBoxSchema,
    DemographicProfileSchema
)
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

# =============================================================================
# Agent Generation
# =============================================================================

def generate_agents(
    scenario: ScenarioType,
    n_agents: int,
    random_state: np.random.RandomState
) -> List[AgentSchema]:
    """
    Generate individual agent data with trust and income attributes.
    
    Args:
        scenario: Policy scenario
        n_agents: Number of agents to generate
        random_state: Random state for reproducibility
        
    Returns:
        List of validated agent schemas
    """
    agents = []
    
    for agent_id in range(n_agents):
        # Generate trust and income from realistic distributions
        trust = random_state.beta(2, 2)  # Beta distribution centered around 0.5
        income = random_state.lognormal(mean=3.5, sigma=0.6)  # Log-normal income
        income = np.clip(income, 0.0, 100.0)
        
        agents.append(AgentSchema(
            agent_id=agent_id,
            trust=trust,
            income=income,
            scenario=scenario
        ))
    
    return agents

from .heatmap_generator import generate_heatmap_grid
from .prim_box_generator import generate_prim_boxes
from .prim_trajectory_generator import generate_prim_trajectory

# =============================================================================
# Demographic Profile Generation (Table III)
# =============================================================================

def generate_demographic_profiles(
    scenario: ScenarioType,
    agents: List[AgentSchema],
    prim_boxes: List[PRIMBoxSchema]
) -> List[DemographicProfileSchema]:
    """
    Generate demographic profile analysis data (Table III).
    
    Args:
        scenario: Policy scenario
        agents: List of agents
        prim_boxes: PRIM boxes identified for scenario
        
    Returns:
        List containing demographic profile for the scenario
    """
    if not prim_boxes:
        raise ValueError("No PRIM boxes provided")
    
    box = prim_boxes[0]
    n_agents_total = len(agents)
    n_agents_segment = int(n_agents_total * box.coverage)
    
    # Generate segment name based on scenario characteristics
    if scenario == ScenarioType.SERVICES_INCENTIVE:
        segment_name = "High Trust Community (Trust ≥ 0.65)"
    elif scenario == ScenarioType.ECONOMIC_INCENTIVE:
        segment_name = "High Trust + Mid-High Income (Trust ≥ 0.55, Income ≥ 30)"
    else:
        segment_name = "Baseline Population (No Segmentation)"
    
    profile = DemographicProfileSchema(
        scenario=scenario,
        segment_name=segment_name,
        trust_min=box.trust_min,
        trust_max=box.trust_max,
        income_min=box.income_min,
        income_max=box.income_max,
        coverage=box.coverage,
        density=box.density,
        lift=box.lift,
        n_agents_total=n_agents_total,
        n_agents_segment=n_agents_segment
    )
    
    return [profile]


# =============================================================================
# NEW: Scale Metadata Generation
# =============================================================================

def generate_scale_metadata(config: GeneratorConfig) -> Dict:
    """
    Generate metadata describing the scales used in the simulation.
    
    Args:
        config: Generator configuration
        
    Returns:
        Dictionary with scale documentation
    """
    metadata = {
        "trust": {
            "original_range": [0.0, 1.0],
            "unit": "normalized_trust_score",
            "binning_method": "uniform",
            "n_bins": config.n_bins,
            "bin_centers": np.linspace(0.025, 0.975, config.n_bins).tolist(),
            "interpretation": "Agent trust propensity score (0=no trust, 1=full trust)",
            "distribution": "Beta(2, 2) - centered around 0.5 with moderate spread"
        },
        "income": {
            "original_range": [0.0, 100.0],
            "unit": "income_percentile",
            "binning_method": "uniform",
            "n_bins": config.n_bins,
            "bin_centers": np.linspace(2.5, 97.5, config.n_bins).tolist(),
            "interpretation": "Income percentile in population (0=lowest, 100=highest)",
            "distribution": "LogNormal(μ=3.5, σ=0.6) clipped to [0, 100]"
        },
        "adoption_rate": {
            "range": [0.0, 1.0],
            "unit": "probability",
            "interpretation": "Proportion of agents adopting green energy technology",
            "uncertainty_quantification": {
                "n_replications": config.n_replications,
                "noise_std": config.noise_std,
                "confidence_level": 0.95
            }
        },
        "generation_metadata": {
            "random_seed": config.random_seed,
            "n_agents": config.n_agents,
            "timestamp": "auto-generated",
            "version": "2.0-enhanced"
        }
    }
    
    return metadata


# =============================================================================
# Orchestration Functions
# =============================================================================

def generate_scenario_data(
    scenario: ScenarioType,
    config: GeneratorConfig
) -> dict:
    """
    Generate all data types for a single scenario.
    
    Args:
        scenario: Policy scenario to generate
        config: Generator configuration
        
    Returns:
        Dictionary with all generated data types
    """
    random_state = np.random.RandomState(config.random_seed)
    
    print(f"\n🔄 Generating data for scenario: {scenario.value}")
    
    # Generate agents
    print(f"  ├─ Generating {config.n_agents} agents...")
    agents = generate_agents(scenario, config.n_agents, random_state)
    
    # Generate heatmap grid WITH replications
    print(f"  ├─ Generating {config.n_bins}×{config.n_bins} heatmap grid "
          f"({config.n_replications} replications)...")
    heatmap_grid, heatmap_replications = generate_heatmap_grid(
        scenario, config.n_bins, config.n_replications, config.noise_std, random_state
    )
    
    # Generate PRIM analysis
    print(f"  ├─ Identifying PRIM boxes...")
    prim_boxes = generate_prim_boxes(scenario, agents, random_state)
    
    # MODIFICATO: Generate PRIM trajectory WITH replications
    print(f"  ├─ Generating PRIM trajectory ({config.n_replications} replications)...")
    prim_trajectory_summary, prim_trajectory_replications = generate_prim_trajectory(
        scenario, 15, config.n_replications, config.noise_std, random_state
    )
    
    print(f"  └─ Generating demographic profiles...")
    demographic_profiles = generate_demographic_profiles(scenario, agents, prim_boxes)
    
    return {
        'agents': agents,
        'heatmap_grid': heatmap_grid,
        'heatmap_replications': heatmap_replications,
        'prim_boxes': prim_boxes,
        'prim_trajectory_summary': prim_trajectory_summary,  # NUOVO
        'prim_trajectory_replications': prim_trajectory_replications,  # NUOVO
        'demographic_profiles': demographic_profiles
    }

def generate_all_scenarios(config: GeneratorConfig) -> dict:
    """
    Generate data for all policy scenarios.
    
    Args:
        config: Generator configuration
        
    Returns:
        Dictionary mapping scenario names to their data
    """
    all_data = {}
    
    scenarios = [
        ScenarioType.NO_INCENTIVE,
        ScenarioType.SERVICES_INCENTIVE,
        ScenarioType.ECONOMIC_INCENTIVE
    ]
    
    for scenario in scenarios:
        all_data[scenario.value] = generate_scenario_data(scenario, config)
    
    return all_data


def save_all_data(all_data: dict, output_dir: Path, config: GeneratorConfig) -> None:
    """
    Save all generated data to CSV files.
    
    Args:
        all_data: Dictionary with all scenario data
        output_dir: Directory to save CSV files
        config: Generator configuration (for metadata)
    """
    from .csv_writer import save_prim_trajectory_summary, save_prim_trajectory_replications
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving data to {output_dir.absolute()}")
    
    # Aggregate data across scenarios
    all_heatmaps = []
    all_heatmap_reps = []
    all_prim_boxes = []
    all_trajectories_summary = []  # MODIFICATO
    all_trajectories_reps = []  # NUOVO
    all_profiles = []
    
    for scenario_name, data in all_data.items():
        all_heatmaps.extend(data['heatmap_grid'])
        all_heatmap_reps.extend(data['heatmap_replications'])
        all_prim_boxes.extend(data['prim_boxes'])
        all_trajectories_summary.extend(data['prim_trajectory_summary'])  # MODIFICATO
        all_trajectories_reps.extend(data['prim_trajectory_replications'])  # NUOVO
        all_profiles.extend(data['demographic_profiles'])
    
    # Save aggregated heatmap (with statistics)
    print("  ├─ Saving heatmap_grid.csv (with CI)...")
    schemas_to_csv(all_heatmaps, output_dir / "heatmap_grid.csv")
    
    # Save disaggregated replications
    print("  ├─ Saving heatmap_replications.csv (all runs)...")
    schemas_to_csv(all_heatmap_reps, output_dir / "heatmap_replications.csv")
    
    # Save other files
    schemas_to_csv(all_prim_boxes, output_dir / "prim_boxes.csv")
    
    # NUOVO: Save PRIM trajectory summary
    print("  ├─ Saving prim_trajectory_summary.csv (with CI)...")
    save_prim_trajectory_summary(all_trajectories_summary, output_dir / "prim_trajectory_summary.csv")
    
    # NUOVO: Save PRIM trajectory replications
    print("  ├─ Saving prim_trajectory_raw.csv (all runs)...")
    save_prim_trajectory_replications(all_trajectories_reps, output_dir / "prim_trajectory_raw.csv")
    
    schemas_to_csv(all_profiles, output_dir / "demographic_profiles.csv")
    
    # Save scale metadata
    print("  └─ Saving scale_metadata.json...")
    metadata = generate_scale_metadata(config)
    with open(output_dir / "scale_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ All data saved successfully!")

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