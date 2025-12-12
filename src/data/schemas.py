"""
Data schemas for ECOGrid montecarlo data generation.

Contains only Pydantic models and enums.
"""

from typing import List
from pydantic import BaseModel, Field, validator
from enum import Enum
from dataclasses import dataclass

# =============================================================================
# Enums
# =============================================================================
class ScenarioType(str, Enum):
    NO_INCENTIVE = "NI"
    SERVICES_INCENTIVE = "SI"
    ECONOMIC_INCENTIVE = "EI"
    MUNICIPAL_TOKENS = "MUNICIPAL_TOKENS"


# =============================================================================
# Schemas
# =============================================================================
class AgentSchema(BaseModel):
    agent_id: int
    trust: float
    income: float
    scenario: ScenarioType
    environmental_concern: float

    class Config:
        use_enum_values = True


class AdoptionRateSchema(BaseModel):
    scenario: ScenarioType
    trust: float
    income: float
    adoption_rate: float
    n_agents: int

    class Config:
        use_enum_values = True


class HeatmapGridSchema(BaseModel):
    scenario: ScenarioType
    trust_bin: float
    income_bin: float
    adoption_rate: float
    n_samples: int

    class Config:
        use_enum_values = True


class PRIMBoxSchema(BaseModel):
    scenario: ScenarioType
    box_id: int
    trust_min: float
    trust_max: float
    income_min: float
    income_max: float
    coverage: float
    density: float
    lift: float

    @validator('trust_max')
    def trust_max_greater_than_min(cls, v, values):
        if 'trust_min' in values and v < values['trust_min']:
            raise ValueError('trust_max must be >= trust_min')
        return v

    @validator('income_max')
    def income_max_greater_than_min(cls, v, values):
        if 'income_min' in values and v < values['income_min']:
            raise ValueError('income_max must be >= income_min')
        return v

    class Config:
        use_enum_values = True


class PRIMTrajectorySchema(BaseModel):
    scenario: ScenarioType
    iteration: int
    coverage: float
    density: float
    n_agents: int
    is_selected: bool = False

    class Config:
        use_enum_values = True

@dataclass
class PRIMTrajectoryReplicationSchema:
    """Individual replication data for PRIM trajectory uncertainty analysis."""
    scenario: ScenarioType
    iteration: int
    replication_id: int
    coverage: float
    density: float
    n_agents: int
    is_selected: bool

@dataclass
class PRIMTrajectoryEnhancedSchema:
    """Enhanced PRIM trajectory with statistical metrics."""
    scenario: ScenarioType
    iteration: int
    coverage_mean: float
    coverage_std: float
    coverage_ci_lower: float
    coverage_ci_upper: float
    density_mean: float
    density_std: float
    density_ci_lower: float
    density_ci_upper: float
    n_agents_mean: float
    is_selected: bool
    n_replications: int

@dataclass
class HeatmapReplicationSchema:
    """Individual replication data for uncertainty analysis."""
    scenario: ScenarioType
    trust_bin: float
    income_bin: float
    replication_id: int
    adoption_rate: float
    n_samples: int

class DemographicProfileSchema(BaseModel):
    scenario: ScenarioType
    segment_name: str
    trust_min: float
    trust_max: float
    income_min: float
    income_max: float
    coverage: float
    density: float
    lift: float
    n_agents_total: int
    n_agents_segment: int

    class Config:
        use_enum_values = True

class HeatmapGridEnhancedSchema(HeatmapGridSchema):
    """Aggregated heatmap grid with statistics."""
    std_dev: float
    ci_lower: float
    ci_upper: float
    n_replications: int