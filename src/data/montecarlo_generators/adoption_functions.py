from typing import Callable
import numpy as np
from src.data.schemas import ScenarioType

def calculate_adoption_no_incentive(trust: float, income: float, noise: float = 0.0) -> float:
    base_rate = 0.15
    trust_effect = 0.15 * trust
    adoption = base_rate + trust_effect + noise
    return np.clip(adoption, 0.0, 1.0)

def calculate_adoption_services_incentive(trust: float, income: float, noise: float = 0.0) -> float:
    base_rate = 0.25
    trust_effect = 0.55 * (trust ** 1.5)
    adoption = base_rate + trust_effect + noise
    return np.clip(adoption, 0.0, 1.0)

def calculate_adoption_economic_incentive(trust: float, income: float, noise: float = 0.0) -> float:
    base_rate = 0.20
    trust_effect = 0.30 * trust
    income_effect = 0.20 * (income / 100.0)
    adoption = base_rate + trust_effect + income_effect + noise
    return np.clip(adoption, 0.0, 1.0)

def get_adoption_function(scenario: ScenarioType) -> Callable[[float, float, float], float]:
    mapping = {
        ScenarioType.NO_INCENTIVE: calculate_adoption_no_incentive,
        ScenarioType.SERVICES_INCENTIVE: calculate_adoption_services_incentive,
        ScenarioType.ECONOMIC_INCENTIVE: calculate_adoption_economic_incentive,
    }
    return mapping[scenario]
