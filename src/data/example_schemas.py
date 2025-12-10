"""
Example usage of ECOGrid schemas.

Generates sample CSV files for verification.
"""

from pathlib import Path
from schemas import (
    AgentSchema,
    AdoptionRateSchema,
    PRIMBoxSchema,
    PRIMTrajectorySchema,
    DemographicProfileSchema,
    ScenarioType,
    schemas_to_csv
)

output_dir = Path("data/montecarlo")
output_dir.mkdir(parents=True, exist_ok=True)

# Sample agents
agents = [
    AgentSchema(agent_id=i, trust=0.5+i*0.1, income=30+i*10, scenario=ScenarioType.NO_INCENTIVE)
    for i in range(5)
]
schemas_to_csv(agents, output_dir / "sample_agents.csv")

# Sample adoption rates
adoption = [
    AdoptionRateSchema(scenario=ScenarioType.SERVICES_INCENTIVE, trust=0.7, income=50, adoption_rate=0.8, n_agents=100),
    AdoptionRateSchema(scenario=ScenarioType.ECONOMIC_INCENTIVE, trust=0.6, income=40, adoption_rate=0.65, n_agents=150)
]
schemas_to_csv(adoption, output_dir / "sample_adoption.csv")

# Sample PRIM box
prim_box = [
    PRIMBoxSchema(scenario=ScenarioType.SERVICES_INCENTIVE, box_id=0, trust_min=0.6, trust_max=1.0, income_min=0, income_max=100, coverage=0.06, density=0.81, lift=2.5)
]
schemas_to_csv(prim_box, output_dir / "sample_prim_box.csv")

# Sample PRIM trajectory
prim_traj = [
    PRIMTrajectorySchema(scenario=ScenarioType.SERVICES_INCENTIVE, iteration=i, coverage=1.0-i*0.1, density=0.3+i*0.05, n_agents=10000-i*1000, is_selected=(i==5))
    for i in range(10)
]
schemas_to_csv(prim_traj, output_dir / "sample_prim_traj.csv")

# Sample demographic profile
demographics = [
    DemographicProfileSchema(scenario=ScenarioType.SERVICES_INCENTIVE, segment_name="High Trust", trust_min=0.7, trust_max=1.0,
                             income_min=0, income_max=100, coverage=0.06, density=0.81, lift=2.5, n_agents_total=10000, n_agents_segment=600)
]
schemas_to_csv(demographics, output_dir / "sample_demographics.csv")

print(f"✅ Sample CSVs created in: {output_dir.absolute()}")
