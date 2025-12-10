from dataclasses import dataclass
from pathlib import Path

@dataclass
class GeneratorConfig:
    """Configuration for montecarlo data generation."""
    n_agents: int = 10000
    n_bins: int = 20
    n_replications: int = 100  # NEW: Number of Monte Carlo replications
    noise_std: float = 0.05
    random_seed: int = 42
    output_dir: Path = Path("data/montecarlo")
