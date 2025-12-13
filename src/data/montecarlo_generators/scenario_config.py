"""
Unified Scenario Configuration Loader.
Loads all scenario parameters from single YAML file.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import yaml
from src.data.schemas import ScenarioType


@dataclass(frozen=True)
class PRIMBoxConfig:
    """PRIM box boundaries configuration."""
    trust_min: float
    trust_max: float
    income_min: float
    income_max: float
    threshold: float
    
    def validate(self) -> None:
        assert 0.0 <= self.trust_min <= self.trust_max <= 1.0
        assert 0.0 <= self.income_min <= self.income_max <= 100.0
        assert 0.0 <= self.threshold <= 1.0


@dataclass(frozen=True)
class AdoptionFunctionConfig:
    """Adoption function parameters configuration."""
    base_rate: float
    trust_coefficient: float
    trust_exponent: float
    income_coefficient: Optional[float]
    income_exponent: Optional[float]
    income_threshold: Optional[float]
    income_multiplier_high: Optional[float]
    income_multiplier_low: Optional[float]
    
    def validate(self) -> None:
        assert 0.0 <= self.base_rate <= 1.0
        assert self.trust_exponent > 0
        if self.income_threshold is not None:
            assert 0.0 <= self.income_threshold <= 1.0


@dataclass(frozen=True)
class PRIMTrajectoryConfig:
    """PRIM trajectory parameters configuration."""
    coverage_start: float
    coverage_end: float
    density_base: float
    density_coefficient: float
    density_exponent: Optional[float]
    selected_iteration_offset: int
    description: str
    
    def validate(self) -> None:
        assert 0.0 <= self.coverage_start <= 1.0
        assert 0.0 <= self.coverage_end <= 1.0
        assert self.coverage_start >= self.coverage_end
        assert 0.0 <= self.density_base <= 1.0


@dataclass(frozen=True)
class GlobalConfig:
    """Global configuration parameters."""
    n_agents_total: int
    coverage_noise_scale: float
    density_noise_scale: float
    
    def validate(self) -> None:
        assert self.n_agents_total > 0
        assert self.coverage_noise_scale >= 0
        assert self.density_noise_scale >= 0


@dataclass(frozen=True)
class ScenarioConfig:
    """Complete scenario configuration."""
    scenario_type: ScenarioType
    description: str
    prim_box: PRIMBoxConfig
    adoption: AdoptionFunctionConfig
    prim_trajectory: PRIMTrajectoryConfig
    
    def validate(self) -> None:
        self.prim_box.validate()
        self.adoption.validate()
        self.prim_trajectory.validate()


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


def _parse_prim_box(data: Dict) -> PRIMBoxConfig:
    """Parse PRIM box configuration."""
    return PRIMBoxConfig(
        trust_min=float(data['trust_min']),
        trust_max=float(data['trust_max']),
        income_min=float(data['income_min']),
        income_max=float(data['income_max']),
        threshold=float(data['threshold'])
    )


def _parse_adoption(data: Dict) -> AdoptionFunctionConfig:
    """Parse adoption function configuration."""
    return AdoptionFunctionConfig(
        base_rate=float(data['base_rate']),
        trust_coefficient=float(data['trust_coefficient']),
        trust_exponent=float(data['trust_exponent']),
        income_coefficient=float(data['income_coefficient']) if data['income_coefficient'] is not None else None,
        income_exponent=float(data['income_exponent']) if data['income_exponent'] is not None else None,
        income_threshold=float(data['income_threshold']) if data['income_threshold'] is not None else None,
        income_multiplier_high=float(data['income_multiplier_high']) if data['income_multiplier_high'] is not None else None,
        income_multiplier_low=float(data['income_multiplier_low']) if data['income_multiplier_low'] is not None else None,
    )


def _parse_prim_trajectory(data: Dict) -> PRIMTrajectoryConfig:
    """Parse PRIM trajectory configuration."""
    return PRIMTrajectoryConfig(
        coverage_start=float(data['coverage_start']),
        coverage_end=float(data['coverage_end']),
        density_base=float(data['density_base']),
        density_coefficient=float(data['density_coefficient']),
        density_exponent=float(data['density_exponent']) if data['density_exponent'] is not None else None,
        selected_iteration_offset=int(data['selected_iteration_offset']),
        description=str(data['description'])
    )


def _parse_global(data: Dict) -> GlobalConfig:
    """Parse global configuration."""
    return GlobalConfig(
        n_agents_total=int(data['n_agents_total']),
        coverage_noise_scale=float(data['noise_scaling']['coverage']),
        density_noise_scale=float(data['noise_scaling']['density'])
    )


def _parse_scenario(scenario_type: ScenarioType, data: Dict) -> ScenarioConfig:
    """Parse complete scenario configuration."""
    return ScenarioConfig(
        scenario_type=scenario_type,
        description=data['description'],
        prim_box=_parse_prim_box(data['prim_box']),
        adoption=_parse_adoption(data['adoption']),
        prim_trajectory=_parse_prim_trajectory(data['prim_trajectory'])
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
        
        # Parse global config
        self._global_config = _parse_global(config_data['global'])
        self._global_config.validate()
        
        # Parse scenario configs
        for scenario, key in SCENARIO_TO_KEY.items():
            scenario_config = _parse_scenario(scenario, config_data['scenarios'][key])
            scenario_config.validate()
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


_loader = ScenarioConfigLoader()


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