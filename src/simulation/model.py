from mesa import Model
from src.simulation.agents.consumer_agent import ConsumerAgent
from src.simulation.agents.prosumer_agent import ProsumerAgent
from src.simulation.agents.grid_agent import GridAgent
import yaml
from pathlib import Path


class SimulationModel(Model):
    """
    Energy Community ABM using Mesa.
    Manages agent creation, configuration, and simulation stepping.
    """
    
    def __init__(self, n_consumers=5, n_prosumers=5, n_grid_agents=1, 
                 scenario="default", config_path=None, seed=None):
        super().__init__(seed=seed)
        
        self.scenario = scenario
        self.n_consumers = n_consumers
        self.n_prosumers = n_prosumers
        self.n_grid_agents = n_grid_agents
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Store grid agent IDs for assignment
        self.grid_agent_ids = []
        
        # Create agents
        self._create_agents()
        
        # Assign grid_id to consumers and prosumers
        self._assign_grid_ids()
    
    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        Falls back to default values if file not found.
        """
        if config_path is None:
            config_path = Path("config/abm_config.yaml")
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default config if file doesn't exist
            print(f"⚠️  Config file not found at {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default configuration values."""
        return {
            'trust': {
                'initial_min': 0.5,
                'initial_max': 1.0,
                'change_min': -0.05,
                'change_max': 0.05,
                'min_value': 0.0,
                'max_value': 1.0
            },
            'income': {
                'consumer': {'min': 0, 'max': 20000},
                'prosumer': {'min': 20000, 'max': 50000},
                'grid': {'value': None}
            },
            'adoption': {
                'initial_min': 0.0,
                'initial_max': 0.0
            },
            'grid': {
                'auto_assign': True
            }
        }
    
    def _create_agents(self):
        """Create all agent instances."""
        agent_id = 0
        
        # Create Consumer agents
        for i in range(self.n_consumers):
            agent = ConsumerAgent(agent_id, self, self.scenario, self.config)
            agent_id += 1
        
        # Create Prosumer agents
        for i in range(self.n_prosumers):
            agent = ProsumerAgent(agent_id, self, self.scenario, self.config)
            agent_id += 1
        
        # Create Grid agents
        for i in range(self.n_grid_agents):
            agent = GridAgent(agent_id, self, self.scenario, self.config)
            self.grid_agent_ids.append(agent_id)
            agent_id += 1
    
    def _assign_grid_ids(self):
        """
        Assign grid_id to consumers and prosumers.
        If auto_assign is True, distribute agents across available grids.
        """
        grid_config = self.config.get('grid', {})
        auto_assign = grid_config.get('auto_assign', True)
        
        if not auto_assign or len(self.grid_agent_ids) == 0:
            return
        
        # Get all non-grid agents
        non_grid_agents = [
            agent for agent in self.agents 
            if not isinstance(agent, GridAgent)
        ]
        
        # Distribute agents across grids (round-robin)
        for idx, agent in enumerate(non_grid_agents):
            grid_idx = idx % len(self.grid_agent_ids)
            agent.grid_id = self.grid_agent_ids[grid_idx]
    
    def step(self):
        """
        Advance the model by one step.
        Agents are shuffled for random activation order.
        """
        for agent in self.agents.shuffle():
            agent.step()
    
    def get_all_agent_states(self):
        """
        Get current state of all agents.
        Returns list of dictionaries suitable for CSV export.
        """
        return [agent.get_state() for agent in self.agents]