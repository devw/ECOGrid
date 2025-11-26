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

from typing import List, Tuple, Callable, Dict
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import sys
import json
from scipy import stats

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.data.schemas import (
    ScenarioType,
    AgentSchema,
    HeatmapGridSchema,
    PRIMBoxSchema,
    PRIMTrajectorySchema,
    DemographicProfileSchema,
    schemas_to_csv
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for dummy data generation."""
    n_agents: int = 10000
    n_bins: int = 20
    n_replications: int = 100  # NEW: Number of Monte Carlo replications
    noise_std: float = 0.05
    random_seed: int = 42
    output_dir: Path = Path("data/dummy")


# =============================================================================
# NEW: Enhanced Schemas for Statistical Metrics
# =============================================================================

@dataclass
class HeatmapGridEnhancedSchema:
    """Enhanced heatmap grid with statistical metrics."""
    scenario: ScenarioType
    trust_bin: float
    income_bin: float
    adoption_rate: float  # Mean across replications
    std_dev: float  # Standard deviation
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    n_replications: int  # Number of Monte Carlo runs
    n_samples: int  # Sample size per bin per replication


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
# Adoption Rate Logic (Scenario-Specific)
# =============================================================================

def calculate_adoption_no_incentive(trust: float, income: float, noise: float = 0.0) -> float:
    """
    Calculate adoption rate for No Incentive (NI) scenario.
    
    Baseline scenario: low adoption, slight trust dependence.
    
    Args:
        trust: Trust level [0, 1]
        income: Income level [0, 100]
        noise: Random noise to add
        
    Returns:
        Adoption rate [0, 1]
    """
    base_rate = 0.15
    trust_effect = 0.15 * trust
    adoption = base_rate + trust_effect + noise
    return np.clip(adoption, 0.0, 1.0)


def calculate_adoption_services_incentive(trust: float, income: float, noise: float = 0.0) -> float:
    """
    Calculate adoption rate for Services Incentive (SI) scenario.
    
    Community-driven: high trust → dramatically higher adoption.
    Income is less relevant.
    
    Args:
        trust: Trust level [0, 1]
        income: Income level [0, 100]
        noise: Random noise to add
        
    Returns:
        Adoption rate [0, 1]
    """
    base_rate = 0.25
    # Strong non-linear trust effect (squared to create concentration)
    trust_effect = 0.55 * (trust ** 1.5)
    adoption = base_rate + trust_effect + noise
    return np.clip(adoption, 0.0, 1.0)


def calculate_adoption_economic_incentive(trust: float, income: float, noise: float = 0.0) -> float:
    """
    Calculate adoption rate for Economic Incentive (EI) scenario.
    
    Mixed drivers: both trust and income matter.
    
    Args:
        trust: Trust level [0, 1]
        income: Income level [0, 100]
        noise: Random noise to add
        
    Returns:
        Adoption rate [0, 1]
    """
    base_rate = 0.20
    trust_effect = 0.30 * trust
    income_effect = 0.20 * (income / 100.0)
    adoption = base_rate + trust_effect + income_effect + noise
    return np.clip(adoption, 0.0, 1.0)


def get_adoption_function(scenario: ScenarioType) -> Callable[[float, float, float], float]:
    """
    Get the appropriate adoption calculation function for a scenario.
    
    Args:
        scenario: Policy scenario type
        
    Returns:
        Function that calculates adoption rate
    """
    mapping = {
        ScenarioType.NO_INCENTIVE: calculate_adoption_no_incentive,
        ScenarioType.SERVICES_INCENTIVE: calculate_adoption_services_incentive,
        ScenarioType.ECONOMIC_INCENTIVE: calculate_adoption_economic_incentive,
    }
    return mapping[scenario]


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


# =============================================================================
# NEW: Enhanced Heatmap Grid Generation with Replications
# =============================================================================

def generate_heatmap_replications(
    scenario: ScenarioType,
    n_bins: int,
    n_replications: int,
    noise_std: float,
    random_state: np.random.RandomState
) -> List[HeatmapReplicationSchema]:
    """
    Generate ALL replications for heatmap grid (disaggregated data).
    
    This creates n_replications × n_bins × n_bins data points for full
    uncertainty quantification and statistical testing.
    
    Args:
        scenario: Policy scenario
        n_bins: Number of bins per dimension
        n_replications: Number of Monte Carlo replications
        noise_std: Standard deviation of noise
        random_state: Random state for reproducibility
        
    Returns:
        List of all replication data points
    """
    adoption_func = get_adoption_function(scenario)
    replications_data = []
    
    trust_bins = np.linspace(0.025, 0.975, n_bins)
    income_bins = np.linspace(2.5, 97.5, n_bins)
    
    for trust in trust_bins:
        for income in income_bins:
            # Generate multiple replications for this grid cell
            for rep_id in range(n_replications):
                noise = random_state.normal(0, noise_std)
                adoption_rate = adoption_func(trust, income, noise)
                
                replications_data.append(HeatmapReplicationSchema(
                    scenario=scenario,
                    trust_bin=trust,
                    income_bin=income,
                    replication_id=rep_id,
                    adoption_rate=adoption_rate,
                    n_samples=500  # Simulated sample size per bin
                ))
    
    return replications_data


def aggregate_replications_to_grid(
    replications: List[HeatmapReplicationSchema]
) -> List[HeatmapGridEnhancedSchema]:
    """
    Aggregate replications into grid with statistical metrics.
    
    Calculates mean, std dev, and 95% confidence intervals for each grid cell.
    
    Args:
        replications: List of all replication data
        
    Returns:
        List of aggregated grid cells with statistics
    """
    # Group by (scenario, trust_bin, income_bin)
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for rep in replications:
        key = (rep.scenario, rep.trust_bin, rep.income_bin)
        grouped[key].append(rep.adoption_rate)
    
    # Calculate statistics for each group
    grid_data = []
    for (scenario, trust, income), rates in grouped.items():
        rates_array = np.array(rates)
        n_reps = len(rates_array)
        
        mean_rate = np.mean(rates_array)
        std_dev = np.std(rates_array, ddof=1)  # Sample std dev
        
        # 95% confidence interval using t-distribution
        ci = stats.t.interval(
            confidence=0.95,
            df=n_reps - 1,
            loc=mean_rate,
            scale=stats.sem(rates_array)
        )
        
        grid_data.append(HeatmapGridEnhancedSchema(
            scenario=scenario,
            trust_bin=trust,
            income_bin=income,
            adoption_rate=mean_rate,
            std_dev=std_dev,
            ci_lower=ci[0],
            ci_upper=ci[1],
            n_replications=n_reps,
            n_samples=500
        ))
    
    return grid_data


def generate_heatmap_grid(
    scenario: ScenarioType,
    n_bins: int,
    n_replications: int,
    noise_std: float,
    random_state: np.random.RandomState
) -> Tuple[List[HeatmapGridEnhancedSchema], List[HeatmapReplicationSchema]]:
    """
    Generate complete heatmap data: both aggregated and disaggregated.
    
    Args:
        scenario: Policy scenario
        n_bins: Number of bins per dimension
        n_replications: Number of Monte Carlo replications
        noise_std: Standard deviation of noise
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (aggregated_grid, all_replications)
    """
    # Generate all replications
    replications = generate_heatmap_replications(
        scenario, n_bins, n_replications, noise_std, random_state
    )
    
    # Aggregate with statistics
    aggregated_grid = aggregate_replications_to_grid(replications)
    
    return aggregated_grid, replications


# =============================================================================
# PRIM Analysis Data Generation (Unchanged)
# =============================================================================

def identify_prim_box(
    scenario: ScenarioType,
    agents: List[AgentSchema],
    adoption_func: Callable[[float, float, float], float]
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Identify PRIM box boundaries for high-adoption segment.
    
    Simulates PRIM algorithm results based on scenario characteristics.
    
    Args:
        scenario: Policy scenario
        agents: List of agents
        adoption_func: Function to calculate adoption
        
    Returns:
        Tuple of (trust_min, trust_max, income_min, income_max, coverage, density, lift)
    """
    if scenario == ScenarioType.SERVICES_INCENTIVE:
        # SI: High trust segment (narrow, high density)
        trust_min, trust_max = 0.65, 1.0
        income_min, income_max = 0.0, 100.0
        coverage = 0.06  # 6% of population
        density = 0.81   # 81% adoption in segment
        lift = 2.5
        
    elif scenario == ScenarioType.ECONOMIC_INCENTIVE:
        # EI: High trust + moderate income (broader segment)
        trust_min, trust_max = 0.55, 1.0
        income_min, income_max = 30.0, 100.0
        coverage = 0.31  # 31% of population
        density = 0.65   # 65% adoption in segment
        lift = 1.8
        
    else:  # NO_INCENTIVE
        # NI: No clear segment (uniform baseline)
        trust_min, trust_max = 0.0, 1.0
        income_min, income_max = 0.0, 100.0
        coverage = 1.0   # 100% (no peeling)
        density = 0.20   # 20% baseline
        lift = 1.0
    
    return trust_min, trust_max, income_min, income_max, coverage, density, lift


def generate_prim_boxes(
    scenario: ScenarioType,
    agents: List[AgentSchema],
    random_state: np.random.RandomState
) -> List[PRIMBoxSchema]:
    """
    Generate PRIM box boundaries for a scenario.
    
    Args:
        scenario: Policy scenario
        agents: List of agents for analysis
        random_state: Random state for reproducibility
        
    Returns:
        List containing the final PRIM box
    """
    adoption_func = get_adoption_function(scenario)
    trust_min, trust_max, income_min, income_max, coverage, density, lift = \
        identify_prim_box(scenario, agents, adoption_func)
    
    box = PRIMBoxSchema(
        scenario=scenario,
        box_id=0,
        trust_min=trust_min,
        trust_max=trust_max,
        income_min=income_min,
        income_max=income_max,
        coverage=coverage,
        density=density,
        lift=lift
    )
    
    return [box]


def generate_prim_trajectory(
    scenario: ScenarioType,
    n_iterations: int,
    random_state: np.random.RandomState
) -> List[PRIMTrajectorySchema]:
    """
    Generate PRIM peeling trajectory data.
    
    Simulates the iterative peeling process showing coverage-density trade-off.
    
    Args:
        scenario: Policy scenario
        n_iterations: Number of peeling iterations
        random_state: Random state for reproducibility
        
    Returns:
        List of trajectory points showing peeling progression
    """
    trajectory = []
    
    if scenario == ScenarioType.SERVICES_INCENTIVE:
        # SI: Dramatic peeling (100% → 6% coverage, maintaining high density)
        coverages = np.linspace(1.0, 0.06, n_iterations)
        # Non-linear density increase
        densities = 0.30 + 0.51 * (1 - coverages) ** 0.8
        selected_iteration = n_iterations - 3  # Near the end
        
    elif scenario == ScenarioType.ECONOMIC_INCENTIVE:
        # EI: Moderate peeling (100% → 31% coverage)
        coverages = np.linspace(1.0, 0.31, n_iterations)
        densities = 0.25 + 0.40 * (1 - coverages) ** 0.6
        selected_iteration = n_iterations - 5
        
    else:  # NO_INCENTIVE
        # NI: Flat trajectory (no meaningful peeling)
        coverages = np.linspace(1.0, 0.95, n_iterations)
        densities = np.full(n_iterations, 0.20) + random_state.normal(0, 0.01, n_iterations)
        selected_iteration = 0  # Select first iteration (no improvement)
    
    for i in range(n_iterations):
        trajectory.append(PRIMTrajectorySchema(
            scenario=scenario,
            iteration=i,
            coverage=float(coverages[i]),
            density=float(np.clip(densities[i], 0, 1)),
            n_agents=int(10000 * coverages[i]),
            is_selected=(i == selected_iteration)
        ))
    
    return trajectory


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
    
    print(f"  ├─ Generating PRIM trajectory...")
    prim_trajectory = generate_prim_trajectory(scenario, 15, random_state)
    
    print(f"  └─ Generating demographic profiles...")
    demographic_profiles = generate_demographic_profiles(scenario, agents, prim_boxes)
    
    return {
        'agents': agents,
        'heatmap_grid': heatmap_grid,
        'heatmap_replications': heatmap_replications,  # NEW
        'prim_boxes': prim_boxes,
        'prim_trajectory': prim_trajectory,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving data to {output_dir.absolute()}")
    
    # Aggregate data across scenarios
    all_heatmaps = []
    all_heatmap_reps = []  # NEW
    all_prim_boxes = []
    all_trajectories = []
    all_profiles = []
    
    for scenario_name, data in all_data.items():
        all_heatmaps.extend(data['heatmap_grid'])
        all_heatmap_reps.extend(data['heatmap_replications'])  # NEW
        all_prim_boxes.extend(data['prim_boxes'])
        all_trajectories.extend(data['prim_trajectory'])
        all_profiles.extend(data['demographic_profiles'])
    
    # Save aggregated heatmap (with statistics)
    print("  ├─ Saving heatmap_grid.csv (with CI)...")
    _save_enhanced_heatmap(all_heatmaps, output_dir / "heatmap_grid.csv")
    
    # NEW: Save disaggregated replications
    print("  ├─ Saving heatmap_replications.csv (all runs)...")
    _save_replications(all_heatmap_reps, output_dir / "heatmap_replications.csv")
    
    # Save other files (unchanged)
    schemas_to_csv(all_prim_boxes, output_dir / "prim_boxes.csv")
    schemas_to_csv(all_trajectories, output_dir / "prim_trajectory.csv")
    schemas_to_csv(all_profiles, output_dir / "demographic_profiles.csv")
    
    # NEW: Save scale metadata
    print("  └─ Saving scale_metadata.json...")
    metadata = generate_scale_metadata(config)
    with open(output_dir / "scale_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ All data saved successfully!")


def _save_enhanced_heatmap(data: List[HeatmapGridEnhancedSchema], filepath: Path):
    """Save enhanced heatmap grid with statistics."""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scenario', 'trust_bin', 'income_bin', 'adoption_rate',
            'std_dev', 'ci_lower', 'ci_upper', 'n_replications', 'n_samples'
        ])
        for item in data:
            writer.writerow([
                item.scenario.value,
                item.trust_bin,
                item.income_bin,
                item.adoption_rate,
                item.std_dev,
                item.ci_lower,
                item.ci_upper,
                item.n_replications,
                item.n_samples
            ])


def _save_replications(data: List[HeatmapReplicationSchema], filepath: Path):
    """Save all individual replications for uncertainty analysis."""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scenario', 'trust_bin', 'income_bin', 'replication_id',
            'adoption_rate', 'n_samples'
        ])
        for item in data:
            writer.writerow([
                item.scenario.value,
                item.trust_bin,
                item.income_bin,
                item.replication_id,
                item.adoption_rate,
                item.n_samples
            ])


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