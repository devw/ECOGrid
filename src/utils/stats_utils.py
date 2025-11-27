#!/usr/bin/env python3
"""
Statistical utilities for uncertainty quantification and confidence intervals.

This module provides functions for:
- Computing descriptive statistics (mean, std, percentiles)
- Calculating confidence intervals (parametric and bootstrap)
- Aggregating replicated simulation data
- Statistical comparison between scenarios

Author: ECOGrid Team
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from scipy import stats
from dataclasses import dataclass


@dataclass
class ConfidenceInterval:
    """Container for confidence interval results."""
    lower: float
    upper: float
    mean: float
    confidence_level: float = 0.95
    
    def __repr__(self) -> str:
        return f"CI({self.confidence_level:.0%}): [{self.lower:.4f}, {self.upper:.4f}] (mean={self.mean:.4f})"


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
    method: str = 'parametric'
) -> ConfidenceInterval:
    """
    Compute confidence interval for a sample.
    
    Args:
        data: Array of observations
        confidence: Confidence level (default 0.95 for 95% CI)
        method: 'parametric' (t-distribution) or 'bootstrap'
    
    Returns:
        ConfidenceInterval object with lower, upper bounds and mean
    
    Examples:
        >>> data = np.random.normal(0.5, 0.1, 100)
        >>> ci = compute_confidence_interval(data)
        >>> print(ci)
        CI(95%): [0.4800, 0.5200] (mean=0.5000)
    """
    if len(data) == 0:
        raise ValueError("Cannot compute CI for empty data")
    
    mean_val = np.mean(data)
    
    if method == 'parametric':
        # Student's t-distribution (appropriate for small samples)
        std_err = stats.sem(data)  # Standard error of the mean
        dof = len(data) - 1  # Degrees of freedom
        ci = stats.t.interval(confidence, dof, loc=mean_val, scale=std_err)
        
    elif method == 'bootstrap':
        # Bootstrap resampling (non-parametric, robust)
        n_bootstrap = 10000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        ci = np.percentile(bootstrap_means, [alpha/2 * 100, (1 - alpha/2) * 100])
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'parametric' or 'bootstrap'")
    
    return ConfidenceInterval(
        lower=ci[0],
        upper=ci[1],
        mean=mean_val,
        confidence_level=confidence
    )


def aggregate_replications(
    df: pd.DataFrame,
    group_cols: List[str],
    value_cols: List[str],
    confidence: float = 0.95,
    ci_method: str = 'parametric'
) -> pd.DataFrame:
    """
    Aggregate replicated simulation data with statistics.
    
    Computes mean, std, and confidence intervals for each group.
    
    Args:
        df: DataFrame with replicated data (must include 'run_id' column)
        group_cols: Columns to group by (e.g., ['scenario', 'iteration'])
        value_cols: Columns to compute statistics for (e.g., ['density', 'coverage'])
        confidence: Confidence level for intervals
        ci_method: Method for CI computation ('parametric' or 'bootstrap')
    
    Returns:
        DataFrame with aggregated statistics:
        - {col}_mean: Mean value
        - {col}_std: Standard deviation
        - {col}_ci_lower: Lower CI bound
        - {col}_ci_upper: Upper CI bound
        - n_replications: Number of runs per group
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'scenario': ['NI']*300,
        ...     'iteration': [0]*100 + [1]*100 + [2]*100,
        ...     'run_id': list(range(100))*3,
        ...     'density': np.random.normal(0.2, 0.05, 300)
        ... })
        >>> result = aggregate_replications(
        ...     df, 
        ...     group_cols=['scenario', 'iteration'],
        ...     value_cols=['density']
        ... )
    """
    if 'run_id' not in df.columns:
        raise ValueError("DataFrame must contain 'run_id' column for aggregation")
    
    grouped = df.groupby(group_cols)
    
    results = []
    for group_key, group_data in grouped:
        group_dict = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        group_dict['n_replications'] = len(group_data)
        
        # Compute statistics for each value column
        for col in value_cols:
            if col not in group_data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            values = group_data[col].values
            
            # Basic statistics
            group_dict[f'{col}_mean'] = np.mean(values)
            group_dict[f'{col}_std'] = np.std(values, ddof=1)  # Sample std
            group_dict[f'{col}_median'] = np.median(values)
            
            # Confidence intervals
            try:
                ci = compute_confidence_interval(values, confidence, ci_method)
                group_dict[f'{col}_ci_lower'] = ci.lower
                group_dict[f'{col}_ci_upper'] = ci.upper
            except Exception as e:
                # Fallback if CI computation fails
                print(f"Warning: CI computation failed for {group_key}, {col}: {e}")
                group_dict[f'{col}_ci_lower'] = np.nan
                group_dict[f'{col}_ci_upper'] = np.nan
        
        results.append(group_dict)
    
    return pd.DataFrame(results)


def compute_standard_error(data: np.ndarray) -> float:
    """
    Compute standard error of the mean.
    
    SE = std / sqrt(n)
    
    Args:
        data: Array of observations
    
    Returns:
        Standard error value
    """
    return stats.sem(data)


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


def compute_percentiles(data: np.ndarray, percentiles: List[float] = [2.5, 25, 50, 75, 97.5]) -> Dict[float, float]:
    """
    Compute multiple percentiles for a dataset.
    
    Args:
        data: Array of observations
        percentiles: List of percentile values (0-100)
    
    Returns:
        Dictionary mapping percentile -> value
    
    Examples:
        >>> data = np.random.normal(0, 1, 1000)
        >>> pcts = compute_percentiles(data)
        >>> print(f"95% CI: [{pcts[2.5]:.2f}, {pcts[97.5]:.2f}]")
    """
    return {p: np.percentile(data, p) for p in percentiles}


# ============================================================================
# CONVENIENCE FUNCTIONS FOR PRIM TRAJECTORY DATA
# ============================================================================

def summarize_trajectory_run(
    df_raw: pd.DataFrame,
    scenario: str,
    iteration: int,
    value_col: str = 'density',
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Summarize statistics for a specific (scenario, iteration) point.
    
    Args:
        df_raw: Raw trajectory data with run_id column
        scenario: Scenario identifier
        iteration: Iteration number
        value_col: Column to summarize
        confidence: Confidence level
    
    Returns:
        Dictionary with statistics
    """
    mask = (df_raw['scenario'] == scenario) & (df_raw['iteration'] == iteration)
    data = df_raw.loc[mask, value_col].values
    
    if len(data) == 0:
        raise ValueError(f"No data found for scenario={scenario}, iteration={iteration}")
    
    ci = compute_confidence_interval(data, confidence)
    
    return {
        'mean': ci.mean,
        'std': np.std(data, ddof=1),
        'ci_lower': ci.lower,
        'ci_upper': ci.upper,
        'n': len(data)
    }


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_replications(
    df: pd.DataFrame,
    group_cols: List[str],
    expected_n_replications: int
) -> pd.DataFrame:
    """
    Validate that all groups have expected number of replications.
    
    Args:
        df: DataFrame with run_id column
        group_cols: Columns defining groups
        expected_n_replications: Expected number of runs per group
    
    Returns:
        DataFrame showing groups with incorrect replication counts
    """
    if 'run_id' not in df.columns:
        raise ValueError("DataFrame must contain 'run_id' column")
    
    counts = df.groupby(group_cols)['run_id'].nunique().reset_index()
    counts.columns = group_cols + ['n_replications']
    
    issues = counts[counts['n_replications'] != expected_n_replications]
    
    if len(issues) > 0:
        print(f"⚠️ Warning: {len(issues)} groups have incorrect replication counts")
        return issues
    else:
        print(f"✅ All groups have correct replication count ({expected_n_replications})")
        return pd.DataFrame()


if __name__ == "__main__":
    # Quick self-test
    print("Testing stats_utils.py...")
    
    # Test 1: Confidence interval
    np.random.seed(42)
    test_data = np.random.normal(0.5, 0.1, 100)
    ci = compute_confidence_interval(test_data)
    print(f"\n✅ Test 1 - Confidence Interval:\n{ci}")
    
    # Test 2: Aggregate replications
    df_test = pd.DataFrame({
        'scenario': ['NI'] * 300,
        'iteration': [0]*100 + [1]*100 + [2]*100,
        'run_id': list(range(100)) * 3,
        'density': np.random.normal(0.2, 0.05, 300)
    })
    
    agg = aggregate_replications(
        df_test,
        group_cols=['scenario', 'iteration'],
        value_cols=['density']
    )
    print(f"\n✅ Test 2 - Aggregation:\n{agg.head()}")
    
    print("\n✅ All tests passed!")