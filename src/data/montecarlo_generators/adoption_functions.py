"""
Adoption Functions - Data-Driven from YAML Config.
Zero hardcoded parameters, fully configurable.
"""
from typing import Callable
import numpy as np
from src.data.schemas import ScenarioType
from .scenario_config import get_adoption_config


def calculate_adoption(
    trust: float,
    income: float,
    scenario: ScenarioType
) -> float:
    """
    Calculate adoption probability for given trust/income using config.
    
    Args:
        trust: Trust level [0, 1]
        income: Income level [0, 100]
        scenario: Policy scenario type
        
    Returns:
        Adoption probability [0, 1]
    """
    config = get_adoption_config(scenario)
    income_norm = income / 100.0
    
    # Base adoption rate
    adoption = config.base_rate
    
    # Trust effect: base_rate + trust_coef * trust^trust_exp
    adoption += config.trust_coefficient * (trust ** config.trust_exponent)
    
    # Income effect: two different strategies based on config
    if config.income_threshold is not None:
        # Multiplier strategy (services_incentive)
        if income_norm > config.income_threshold:
            income_multiplier = 1.0 + config.income_multiplier_high * income_norm
        else:
            income_multiplier = 1.0 + config.income_multiplier_low * income_norm
        adoption *= income_multiplier
    else:
        # Linear coefficient strategy (no_incentive, economic_incentive)
        income_effect = config.income_coefficient * (income_norm ** config.income_exponent)
        adoption += income_effect
    
    return np.clip(adoption, 0.0, 1.0)


def get_adoption_function(scenario: ScenarioType) -> Callable[[float, float], float]:
    """
    Get adoption function for scenario.
    Returns a callable that takes (trust, income) and returns adoption probability.
    """
    def adoption_func(trust: float, income: float) -> float:
        return calculate_adoption(trust, income, scenario)
    
    return adoption_func