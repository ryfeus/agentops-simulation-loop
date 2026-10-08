"""Shared model configuration for public contracts."""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Strict contract base that rejects fields not declared by the schema."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
