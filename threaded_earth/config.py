from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from threaded_earth.paths import DEFAULT_CONFIG_PATH


class SettlementConfig(BaseModel):
    location_id: str
    name: str
    symbolic_description: str


class StartingResourcesConfig(BaseModel):
    grain_per_household: int = 18
    fish_per_household: int = 6
    reeds_per_household: int = 5
    tools_per_household: int = 3


class HouseholdConfig(BaseModel):
    min_size: int = 3
    max_size: int = 6


class DecisionConfig(BaseModel):
    conflict_threshold: float = 0.72
    cooperation_threshold: float = 0.58
    memory_salience_threshold: float = 0.62


class SimulationConfig(BaseModel):
    population_size: int = 50
    settlement: SettlementConfig
    starting_resources: StartingResourcesConfig = Field(default_factory=StartingResourcesConfig)
    household: HouseholdConfig = Field(default_factory=HouseholdConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)


class ThreadedEarthConfig(BaseModel):
    simulation: SimulationConfig
    archetypes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ThreadedEarthConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ThreadedEarthConfig.model_validate(data)
