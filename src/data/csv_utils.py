"""
Utilities for converting Pydantic schemas to/from CSV.
"""

from typing import List
from pydantic import BaseModel
import pandas as pd


def schemas_to_csv(schemas: List[BaseModel], output_path: str) -> None:
    """
    Convert list of Pydantic schemas to a CSV file.
    """
    if not schemas:
        raise ValueError("Empty schema list provided")
    df = pd.DataFrame([s.dict() for s in schemas])
    df.to_csv(output_path, index=False)


def csv_to_schemas(csv_path: str, schema_class: type[BaseModel]) -> List[BaseModel]:
    """
    Load CSV and convert rows to instances of a Pydantic schema.
    """
    df = pd.read_csv(csv_path)
    return [schema_class(**row.to_dict()) for _, row in df.iterrows()]
