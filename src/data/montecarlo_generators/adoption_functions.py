from typing import Callable
import numpy as np
from src.data.schemas import ScenarioType


def calculate_adoption_no_incentive(trust: float, income: float, noise: float = 0.0) -> float:
    base_rate = 0.05
    income_effect = 0.20 * (income / 100.0)
    trust_effect = 0.10 * trust
    
    adoption = base_rate + income_effect + trust_effect + noise
    return np.clip(adoption, 0.0, 1.0)


def calculate_adoption_services_incentive(trust: float, income: float, noise: float = 0.0) -> float:
    base_rate = 0.08
    trust_effect = 0.22 * (trust ** 1.8)
    
    # Non-linear income effect: threshold at 70th percentile
    income_norm = income / 100.0
    if income_norm > 0.7:
        # High income: strong multiplier (community services adoption)
        income_multiplier = 1.0 + 1.6 * income_norm
    else:
        # Low/Mid income: moderate multiplier
        income_multiplier = 1.0 + 0.5 * income_norm
    
    adoption = (base_rate + trust_effect) * income_multiplier + noise
    return np.clip(adoption, 0.0, 1.0)


def calculate_adoption_economic_incentive(trust: float, income: float, noise: float = 0.0) -> float:
    base_rate = 0.28
    trust_effect = 0.15 * trust
    
    # Quadratic income effect (U-shaped: low and high benefit slightly more)
    income_normalized = income / 100.0
    income_effect = 0.08 * (income_normalized ** 2 - income_normalized + 0.5)
    
    adoption = base_rate + trust_effect + income_effect + noise
    return np.clip(adoption, 0.0, 1.0)


def get_adoption_function(scenario: ScenarioType) -> Callable[[float, float, float], float]:
    """Get calibrated adoption function for a given scenario."""
    mapping = {
        ScenarioType.NO_INCENTIVE: calculate_adoption_no_incentive,
        ScenarioType.SERVICES_INCENTIVE: calculate_adoption_services_incentive,
        ScenarioType.ECONOMIC_INCENTIVE: calculate_adoption_economic_incentive,
    }
    return mapping[scenario]