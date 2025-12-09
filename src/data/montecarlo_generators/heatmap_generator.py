"""
Heatmap generation functions for ECOGrid Monte Carlo simulation.
Refactored from calibrated_generator.py
"""

from typing import List
import numpy as np
from scipy import stats

from src.data.schemas import (
    ScenarioType,
    HeatmapGridSchema,
    HeatmapGridSchema
)
from src.data.montecarlo_generators.adoption_functions import get_adoption_function


def generate_heatmap_replications(
    scenario: ScenarioType,
    n_bins: int,
    n_replications: int,
    noise_std: float,
    random_state: np.random.RandomState
) -> List[HeatmapGridSchema]:
    """
    Generate ALL replications for heatmap grid (disaggregated data).
    """
    adoption_func = get_adoption_function(scenario)
    replications_data = []

    trust_bins = np.linspace(0.025, 0.975, n_bins)
    income_bins = np.linspace(2.5, 97.5, n_bins)

    for trust in trust_bins:
        for income in income_bins:
            for rep_id in range(n_replications):
                noise = random_state.normal(0, noise_std)
                adoption_rate = adoption_func(trust, income, noise)
                replications_data.append(HeatmapGridSchema(
                    scenario=scenario,
                    trust_bin=trust,
                    income_bin=income,
                    replication_id=rep_id,
                    adoption_rate=adoption_rate,
                    n_samples=500
                ))

    return replications_data


def aggregate_replications_to_grid(
    replications: List[HeatmapGridSchema]
) -> List[HeatmapGridSchema]:
    """
    Aggregate replications into grid with statistical metrics.
    """
    from collections import defaultdict
    grouped = defaultdict(list)

    for rep in replications:
        key = (rep.scenario, rep.trust_bin, rep.income_bin)
        grouped[key].append(rep.adoption_rate)

    grid_data = []
    for (scenario, trust, income), rates in grouped.items():
        rates_array = np.array(rates)
        n_reps = len(rates_array)

        mean_rate = np.mean(rates_array)
        std_dev = np.std(rates_array, ddof=1)
        ci = stats.t.interval(
            confidence=0.95,
            df=n_reps - 1,
            loc=mean_rate,
            scale=stats.sem(rates_array)
        )

    grid_data.append(HeatmapGridSchema(
        scenario=scenario,
        trust_bin=trust,
        income_bin=income,
        adoption_rate=mean_rate,
        n_samples=500
    ))

    return grid_data


def generate_heatmap_grid(
    scenario: ScenarioType,
    n_bins: int,
    n_replications: int,
    noise_std: float,
    random_state: np.random.RandomState
) -> tuple[list[HeatmapGridSchema], list[HeatmapGridSchema]]:
    """
    Generate complete heatmap data: aggregated + disaggregated.
    """
    replications = generate_heatmap_replications(
        scenario, n_bins, n_replications, noise_std, random_state
    )
    aggregated_grid = aggregate_replications_to_grid(replications)
    return aggregated_grid, replications
