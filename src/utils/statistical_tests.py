"""
Statistical hypothesis testing utilities.

This module provides functions for comparing scenarios and validating
statistical assumptions in agent-based model simulations.
"""

from typing import Dict
import numpy as np
import pandas as pd
from scipy import stats


def compare_scenarios_ttest(
    df: pd.DataFrame,
    scenario_col: str,
    value_col: str,
    scenario_a: str,
    scenario_b: str
) -> Dict[str, float]:
    """
    Perform t-test comparison between two scenarios.
    
    Args:
        df: DataFrame with replicated data
        scenario_col: Column name for scenario identifier
        value_col: Column name for value to compare
        scenario_a: First scenario identifier
        scenario_b: Second scenario identifier
    
    Returns:
        Dictionary with test results:
        - statistic: t-statistic
        - pvalue: p-value
        - mean_a: Mean of scenario A
        - mean_b: Mean of scenario B
        - significant: Whether difference is significant (p < 0.05)
    
    Examples:
        >>> result = compare_scenarios_ttest(
        ...     df, 'scenario', 'density', 'NI', 'SI'
        ... )
        >>> if result['significant']:
        ...     print(f"Significant difference: p={result['pvalue']:.4f}")
    """
    data_a = df[df[scenario_col] == scenario_a][value_col].values
    data_b = df[df[scenario_col] == scenario_b][value_col].values
    
    if len(data_a) == 0 or len(data_b) == 0:
        raise ValueError(f"No data found for scenarios {scenario_a} or {scenario_b}")
    
    # Welch's t-test (doesn't assume equal variances)
    statistic, pvalue = stats.ttest_ind(data_a, data_b, equal_var=False)
    
    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'mean_a': np.mean(data_a),
        'mean_b': np.mean(data_b),
        'std_a': np.std(data_a, ddof=1),
        'std_b': np.std(data_b, ddof=1),
        'significant': pvalue < 0.05,
        'effect_size': (np.mean(data_a) - np.mean(data_b)) / np.sqrt((np.std(data_a)**2 + np.std(data_b)**2) / 2)  # Cohen's d
    }


def check_normality(data: np.ndarray, alpha: float = 0.05) -> Dict[str, any]:
    """
    Check if data follows normal distribution using Shapiro-Wilk test.
    
    Args:
        data: Array of observations
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    if len(data) < 3:
        return {'normal': None, 'warning': 'Sample too small for normality test'}
    
    statistic, pvalue = stats.shapiro(data)
    
    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'normal': pvalue > alpha,
        'message': 'Normal' if pvalue > alpha else 'Not normal (consider bootstrap CI)'
    }