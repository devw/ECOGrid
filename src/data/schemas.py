"""
Data schemas for ECOGrid montecarlo data generation.

This module defines the structure of all data types used in the simulation
using Pydantic for validation and easy CSV conversion.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class ScenarioType(str, Enum):
    """Policy scenario types."""
    NO_INCENTIVE = "NI"
    SERVICES_INCENTIVE = "SI"
    ECONOMIC_INCENTIVE = "EI"


class AgentSchema(BaseModel):
    """Schema for individual agent data."""
    agent_id: int = Field(..., description="Unique agent identifier")
    trust: float = Field(..., ge=0.0, le=1.0, description="Trust level (0-1)")
    income: float = Field(..., ge=0.0, le=100.0, description="Income level (0-100)")
    scenario: ScenarioType = Field(..., description="Policy scenario")
    
    class Config:
        use_enum_values = True


class AdoptionRateSchema(BaseModel):
    """Schema for adoption rate data by scenario."""
    scenario: ScenarioType = Field(..., description="Policy scenario")
    trust: float = Field(..., ge=0.0, le=1.0, description="Trust level")
    income: float = Field(..., ge=0.0, le=100.0, description="Income level")
    adoption_rate: float = Field(..., ge=0.0, le=1.0, description="Adoption rate (0-1)")
    n_agents: int = Field(..., gt=0, description="Number of agents in this bin")
    
    class Config:
        use_enum_values = True


class HeatmapGridSchema(BaseModel):
    """Schema for heatmap grid data (Trust x Income)."""
    scenario: ScenarioType = Field(..., description="Policy scenario")
    trust_bin: float = Field(..., ge=0.0, le=1.0, description="Trust bin center")
    income_bin: float = Field(..., ge=0.0, le=100.0, description="Income bin center")
    adoption_rate: float = Field(..., ge=0.0, le=1.0, description="Average adoption rate")
    n_samples: int = Field(..., ge=0, description="Number of samples in bin")
    
    class Config:
        use_enum_values = True


class PRIMBoxSchema(BaseModel):
    """Schema for PRIM box boundaries."""
    scenario: ScenarioType = Field(..., description="Policy scenario")
    box_id: int = Field(..., ge=0, description="Box identifier")
    trust_min: float = Field(..., ge=0.0, le=1.0, description="Minimum trust")
    trust_max: float = Field(..., ge=0.0, le=1.0, description="Maximum trust")
    income_min: float = Field(..., ge=0.0, le=100.0, description="Minimum income")
    income_max: float = Field(..., ge=0.0, le=100.0, description="Maximum income")
    coverage: float = Field(..., ge=0.0, le=1.0, description="Population coverage")
    density: float = Field(..., ge=0.0, le=1.0, description="Adoption density")
    lift: float = Field(..., gt=0.0, description="Lift over baseline")
    
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
    """Schema for PRIM peeling trajectory data."""
    scenario: ScenarioType = Field(..., description="Policy scenario")
    iteration: int = Field(..., ge=0, description="Peeling iteration number")
    coverage: float = Field(..., ge=0.0, le=1.0, description="Population coverage")
    density: float = Field(..., ge=0.0, le=1.0, description="Adoption density")
    n_agents: int = Field(..., ge=0, description="Number of agents in box")
    is_selected: bool = Field(default=False, description="Is this the selected box?")
    
    class Config:
        use_enum_values = True


class DemographicProfileSchema(BaseModel):
    """Schema for demographic profile analysis (Table III)."""
    scenario: ScenarioType = Field(..., description="Policy scenario")
    segment_name: str = Field(..., description="Segment description")
    trust_min: float = Field(..., ge=0.0, le=1.0, description="Minimum trust")
    trust_max: float = Field(..., ge=0.0, le=1.0, description="Maximum trust")
    income_min: float = Field(..., ge=0.0, le=100.0, description="Minimum income")
    income_max: float = Field(..., ge=0.0, le=100.0, description="Maximum income")
    coverage: float = Field(..., ge=0.0, le=1.0, description="Population coverage %")
    density: float = Field(..., ge=0.0, le=1.0, description="Adoption rate %")
    lift: float = Field(..., gt=0.0, description="Lift over baseline")
    n_agents_total: int = Field(..., gt=0, description="Total agents analyzed")
    n_agents_segment: int = Field(..., ge=0, description="Agents in segment")
    
    class Config:
        use_enum_values = True


# Utility functions for CSV conversion
def schemas_to_csv(schemas: List[BaseModel], output_path: str) -> None:
    """
    Convert list of Pydantic schemas to CSV file.
    
    Args:
        schemas: List of Pydantic model instances
        output_path: Path to output CSV file
    """
    import pandas as pd
    
    if not schemas:
        raise ValueError("Empty schema list provided")
    
    # Convert to dictionaries
    data = [schema.dict() for schema in schemas]
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(schemas)} records to {output_path}")


def csv_to_schemas(csv_path: str, schema_class: type[BaseModel]) -> List[BaseModel]:
    """
    Load CSV file and convert to Pydantic schemas.
    
    Args:
        csv_path: Path to CSV file
        schema_class: Pydantic model class to instantiate
        
    Returns:
        List of validated schema instances
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Convert each row to schema instance
    schemas = []
    for _, row in df.iterrows():
        schema = schema_class(**row.to_dict())
        schemas.append(schema)
    
    print(f"✅ Loaded {len(schemas)} records from {csv_path}")
    return schemas


# Example usage for testing
if __name__ == "__main__":
    """
    Test schema creation and CSV generation.
    Run this to verify schemas work correctly.
    """
    import os
    from pathlib import Path
    
    # Create output directory
    output_dir = Path("data/montecarlo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧪 Testing schemas...")
    
    # Test 1: Create sample agents
    print("\n1️⃣ Creating sample agents...")
    sample_agents = [
        AgentSchema(
            agent_id=i,
            trust=0.5 + i * 0.1,
            income=30.0 + i * 10.0,
            scenario=ScenarioType.NO_INCENTIVE
        )
        for i in range(5)
    ]
    
    # Save to CSV
    schemas_to_csv(sample_agents, output_dir / "test_agents.csv")
    
    # Test 2: Create sample adoption rates
    print("\n2️⃣ Creating sample adoption rates...")
    sample_adoption = [
        AdoptionRateSchema(
            scenario=ScenarioType.SERVICES_INCENTIVE,
            trust=0.7,
            income=50.0,
            adoption_rate=0.8,
            n_agents=100
        ),
        AdoptionRateSchema(
            scenario=ScenarioType.ECONOMIC_INCENTIVE,
            trust=0.6,
            income=40.0,
            adoption_rate=0.65,
            n_agents=150
        )
    ]
    
    schemas_to_csv(sample_adoption, output_dir / "test_adoption_rates.csv")
    
    # Test 3: Create sample PRIM box
    print("\n3️⃣ Creating sample PRIM box...")
    sample_prim_box = [
        PRIMBoxSchema(
            scenario=ScenarioType.SERVICES_INCENTIVE,
            box_id=0,
            trust_min=0.6,
            trust_max=1.0,
            income_min=0.0,
            income_max=100.0,
            coverage=0.06,
            density=0.81,
            lift=2.5
        )
    ]
    
    schemas_to_csv(sample_prim_box, output_dir / "test_prim_boxes.csv")
    
    # Test 4: Create sample trajectory
    print("\n4️⃣ Creating sample PRIM trajectory...")
    sample_trajectory = [
        PRIMTrajectorySchema(
            scenario=ScenarioType.SERVICES_INCENTIVE,
            iteration=i,
            coverage=1.0 - i * 0.1,
            density=0.3 + i * 0.05,
            n_agents=10000 - i * 1000,
            is_selected=(i == 5)
        )
        for i in range(10)
    ]
    
    schemas_to_csv(sample_trajectory, output_dir / "test_prim_trajectory.csv")
    
    # Test 5: Create sample demographic profile
    print("\n5️⃣ Creating sample demographic profile...")
    sample_profile = [
        DemographicProfileSchema(
            scenario=ScenarioType.SERVICES_INCENTIVE,
            segment_name="High Trust Segment",
            trust_min=0.7,
            trust_max=1.0,
            income_min=0.0,
            income_max=100.0,
            coverage=0.06,
            density=0.81,
            lift=2.5,
            n_agents_total=10000,
            n_agents_segment=600
        )
    ]
    
    schemas_to_csv(sample_profile, output_dir / "test_demographic_profiles.csv")
    
    print("\n✅ All schemas tested successfully!")
    print(f"📁 Test CSV files created in: {output_dir.absolute()}")
    print("\n📊 You can now analyze these files to verify structure:")
    print(f"   - {output_dir / 'test_agents.csv'}")
    print(f"   - {output_dir / 'test_adoption_rates.csv'}")
    print(f"   - {output_dir / 'test_prim_boxes.csv'}")
    print(f"   - {output_dir / 'test_prim_trajectory.csv'}")
    print(f"   - {output_dir / 'test_demographic_profiles.csv'}")