"""
Statistical utilities for Monte Carlo simulations.

This module provides functions for aggregating replicated simulation data
with confidence intervals and uncertainty quantification.
"""

from typing import List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


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