"""
Unified Scenario Configuration Loader - Refactored with Pydantic.
Loads all scenario parameters from single YAML file with cleaner validation.
"""
from pathlib import Path
from typing import Dict, Optional
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from src.data.schemas import ScenarioType


class PRIMBoxConfig(BaseModel):
    """PRIM box boundaries configuration."""
    trust_min: float = Field(ge=0.0, le=1.0)
    trust_max: float = Field(default=1.0, ge=0.0, le=1.0)
    income_min: float = Field(ge=0.0, le=100.0)
    income_max: float = Field(ge=0.0, le=100.0)
    threshold: float = Field(ge=0.0, le=1.0)
    
    @model_validator(mode='after')
    def validate_ranges(self):
        if self.trust_max < self.trust_min:
            raise ValueError(f'trust_max ({self.trust_max}) < trust_min ({self.trust_min})')
        if self.income_max < self.income_min:
            raise ValueError(f'income_max ({self.income_max}) < income_min ({self.income_min})')
        return self


class AdoptionFunctionConfig(BaseModel):
    """Adoption function parameters configuration."""
    base_rate: float = Field(ge=0.0, le=1.0)
    trust_coefficient: float
    trust_exponent: float = Field(gt=0.0)
    income_coefficient: Optional[float] = None
    income_exponent: Optional[float] = None
    income_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    income_multiplier_high: Optional[float] = None
    income_multiplier_low: Optional[float] = None


class PRIMTrajectoryConfig(BaseModel):
    """PRIM trajectory parameters configuration."""
    coverage_start: float = Field(default=1.0, ge=0.0, le=1.0)
    coverage_end: float = Field(ge=0.0, le=1.0)
    density_base: float = Field(ge=0.0, le=1.0)
    density_coefficient: float
    density_exponent: Optional[float] = None
    selected_iteration_offset: int
    description: str
    
    @model_validator(mode='after')
    def validate_coverage_range(self):
        if self.coverage_start < self.coverage_end:
            raise ValueError(
                f'coverage_start ({self.coverage_start}) < coverage_end ({self.coverage_end})'
            )
        return self


class NoiseScalingConfig(BaseModel):
    """Noise scaling configuration."""
    coverage: float = Field(ge=0.0)
    density: float = Field(ge=0.0)


class GlobalConfig(BaseModel):
    """Global configuration parameters."""
    n_agents_total: int = Field(gt=0)
    noise_scaling: NoiseScalingConfig

    @property
    def coverage_noise_scale(self) -> float:
        """Backward compatibility property."""
        return self.noise_scaling.coverage
    
    @property
    def density_noise_scale(self) -> float:
        """Backward compatibility property."""
        return self.noise_scaling.density

class DefaultsConfig(BaseModel):
    """Default values applied to all scenarios."""
    trust_max: float = Field(default=1.0, ge=0.0, le=1.0)
    coverage_start: float = Field(default=1.0, ge=0.0, le=1.0)
    n_agents_total: int = Field(gt=0)
    noise_scaling: NoiseScalingConfig


class ScenarioConfig(BaseModel):
    """Complete scenario configuration."""
    scenario_type: ScenarioType
    description: str
    prim_box: PRIMBoxConfig
    adoption: AdoptionFunctionConfig
    prim_trajectory: PRIMTrajectoryConfig


SCENARIO_TO_KEY = {
    ScenarioType.NO_INCENTIVE: 'no_incentive',
    ScenarioType.SERVICES_INCENTIVE: 'services_incentive',
    ScenarioType.ECONOMIC_INCENTIVE: 'economic_incentive',
}


def _load_yaml_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _apply_defaults(scenario_data: Dict, defaults: DefaultsConfig) -> Dict:
    """Apply default values to scenario data."""
    scenario_data = scenario_data.copy()
    
    # Apply to prim_box
    if 'prim_box' in scenario_data:
        scenario_data['prim_box'].setdefault('trust_max', defaults.trust_max)
    
    # Apply to prim_trajectory
    if 'prim_trajectory' in scenario_data:
        scenario_data['prim_trajectory'].setdefault('coverage_start', defaults.coverage_start)
    
    return scenario_data


def _parse_scenario(scenario_type: ScenarioType, data: Dict, defaults: DefaultsConfig) -> ScenarioConfig:
    """Parse complete scenario configuration with defaults applied."""
    data = _apply_defaults(data, defaults)
    
    return ScenarioConfig(
        scenario_type=scenario_type,
        description=data['description'],
        prim_box=PRIMBoxConfig(**data['prim_box']),
        adoption=AdoptionFunctionConfig(**data['adoption']),
        prim_trajectory=PRIMTrajectoryConfig(**data['prim_trajectory'])
    )


class ScenarioConfigLoader:
    """Singleton loader for unified scenario configurations."""
    
    _instance = None
    _config_cache: Dict[ScenarioType, ScenarioConfig] = {}
    _global_config: Optional[GlobalConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, config_path: Path = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parents[3] / 'config' / 'scenario_parameters.yaml'
        
        config_data = _load_yaml_config(config_path)
        
        # Parse defaults (backward compatible with 'global' key)
        defaults_data = config_data.get('defaults', config_data.get('global', {}))
        defaults = DefaultsConfig(**defaults_data)
        
        # Create global config from defaults
        self._global_config = GlobalConfig(
            n_agents_total=defaults.n_agents_total,
            noise_scaling=defaults.noise_scaling
        )
        
        # Parse scenario configs
        for scenario, key in SCENARIO_TO_KEY.items():
            scenario_config = _parse_scenario(scenario, config_data['scenarios'][key], defaults)
            self._config_cache[scenario] = scenario_config
    
    def get_config(self, scenario: ScenarioType) -> ScenarioConfig:
        """Get complete configuration for scenario."""
        if not self._config_cache:
            self.load_config()
        
        if scenario not in self._config_cache:
            raise ValueError(f"No configuration found for scenario: {scenario}")
        
        return self._config_cache[scenario]
    
    def get_prim_box(self, scenario: ScenarioType) -> PRIMBoxConfig:
        """Get PRIM box configuration for scenario."""
        return self.get_config(scenario).prim_box
    
    def get_adoption(self, scenario: ScenarioType) -> AdoptionFunctionConfig:
        """Get adoption function configuration for scenario."""
        return self.get_config(scenario).adoption
    
    def get_prim_trajectory(self, scenario: ScenarioType) -> PRIMTrajectoryConfig:
        """Get PRIM trajectory configuration for scenario."""
        return self.get_config(scenario).prim_trajectory
    
    def get_global(self) -> GlobalConfig:
        """Get global configuration."""
        if self._global_config is None:
            self.load_config()
        return self._global_config
    
    def reload(self, config_path: Path = None) -> None:
        """Force reload configuration from file."""
        self._config_cache.clear()
        self._global_config = None
        self.load_config(config_path)


# Singleton instance
_loader = ScenarioConfigLoader()


# Public API functions
def get_scenario_config(scenario: ScenarioType) -> ScenarioConfig:
    """Get complete scenario configuration."""
    return _loader.get_config(scenario)


def get_prim_box_config(scenario: ScenarioType) -> PRIMBoxConfig:
    """Get PRIM box configuration."""
    return _loader.get_prim_box(scenario)


def get_adoption_config(scenario: ScenarioType) -> AdoptionFunctionConfig:
    """Get adoption function configuration."""
    return _loader.get_adoption(scenario)


def get_prim_trajectory_config(scenario: ScenarioType) -> PRIMTrajectoryConfig:
    """Get PRIM trajectory configuration."""
    return _loader.get_prim_trajectory(scenario)


def get_global_config() -> GlobalConfig:
    """Get global configuration."""
    return _loader.get_global()


def reload_config(config_path: Path = None) -> None:
    """Reload configuration from file."""
    _loader.reload(config_path)