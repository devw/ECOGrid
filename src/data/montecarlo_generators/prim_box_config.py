"""
PRIM Box Configuration Loader.
Loads boundaries from YAML config file.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import yaml
from src.data.schemas import ScenarioType


@dataclass(frozen=True)
class PRIMBoxBoundaries:
    """Immutable PRIM box boundaries."""
    trust_min: float
    trust_max: float
    income_min: float
    income_max: float
    threshold: float
    
    def validate(self) -> None:
        """Validate boundaries are within acceptable ranges."""
        assert 0.0 <= self.trust_min <= self.trust_max <= 1.0, \
            f"Invalid trust range: [{self.trust_min}, {self.trust_max}]"
        assert 0.0 <= self.income_min <= self.income_max <= 100.0, \
            f"Invalid income range: [{self.income_min}, {self.income_max}]"
        assert 0.0 <= self.threshold <= 1.0, \
            f"Invalid threshold: {self.threshold}"


def _load_yaml_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _parse_boundaries(config_data: Dict, scenario_key: str) -> PRIMBoxBoundaries:
    """Parse boundaries from config data."""
    scenario_config = config_data['scenarios'][scenario_key]
    
    return PRIMBoxBoundaries(
        trust_min=float(scenario_config['trust_min']),
        trust_max=float(scenario_config['trust_max']),
        income_min=float(scenario_config['income_min']),
        income_max=float(scenario_config['income_max']),
        threshold=float(scenario_config['threshold'])
    )


# Scenario enum to config key mapping
SCENARIO_TO_KEY = {
    ScenarioType.NO_INCENTIVE: 'no_incentive',
    ScenarioType.SERVICES_INCENTIVE: 'services_incentive',
    ScenarioType.ECONOMIC_INCENTIVE: 'economic_incentive',
}


class PRIMConfigLoader:
    """Singleton loader for PRIM box configurations."""
    
    _instance = None
    _config_cache: Dict[ScenarioType, PRIMBoxBoundaries] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, config_path: Path = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default path: config/prim_boxes.yaml
            config_path = Path(__file__).parents[3] / 'config' / 'prim_boxes.yaml'
        
        config_data = _load_yaml_config(config_path)
        
        # Parse and cache all scenarios
        for scenario, key in SCENARIO_TO_KEY.items():
            boundaries = _parse_boundaries(config_data, key)
            boundaries.validate()
            self._config_cache[scenario] = boundaries
    
    def get_boundaries(self, scenario: ScenarioType) -> PRIMBoxBoundaries:
        """
        Get boundaries for scenario.
        Lazy loads config on first call.
        """
        if not self._config_cache:
            self.load_config()
        
        if scenario not in self._config_cache:
            raise ValueError(f"No configuration found for scenario: {scenario}")
        
        return self._config_cache[scenario]
    
    def reload(self, config_path: Path = None) -> None:
        """Force reload configuration from file."""
        self._config_cache.clear()
        self.load_config(config_path)


# Global loader instance
_loader = PRIMConfigLoader()


def get_prim_boundaries(scenario: ScenarioType) -> PRIMBoxBoundaries:
    """
    Get PRIM box boundaries for a scenario.
    
    Loads from config/prim_boxes.yaml on first call.
    
    Args:
        scenario: Policy scenario type
        
    Returns:
        Immutable boundaries configuration
    """
    return _loader.get_boundaries(scenario)


def reload_config(config_path: Path = None) -> None:
    """
    Reload configuration from file.
    Useful for testing or runtime updates.
    """
    _loader.reload(config_path)