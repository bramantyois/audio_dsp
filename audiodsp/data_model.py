from pydantic import BaseModel
from typing import Optional, Dict


class Parameters(BaseModel):
    name: str
    module_name: str
    values: Optional[Dict[str, float]] = None


class Coefficients(BaseModel):
    name: str
    module_name: str
    values: Optional[Dict[str, float]] = None