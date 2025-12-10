"""
Metadata Generation for Agent-Based Model.

This module generates metadata describing the scales and parameters
used in Monte Carlo simulations for energy transition scenarios.
"""

from typing import Dict
import numpy as np

from .config import GeneratorConfig


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