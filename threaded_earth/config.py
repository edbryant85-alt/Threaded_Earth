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


class UpkeepConfig(BaseModel):
    daily_food_need_per_agent: float = 1.0
    food_shortage_memory_threshold: float = 1.0
    material_decay_enabled: bool = False
    material_decay_per_household: float = 0.0


class PropagationConfig(BaseModel):
    propagation_enabled: bool = True
    propagation_max_observers: int = 5
    propagation_max_events_per_tick: int = 80
    propagation_max_memories_per_tick: int = 80
    propagation_min_source_event_importance: float = 0.55
    propagation_cooldown_per_observer_subject_pair: bool = False
    propagation_strength: float = 0.2
    propagation_min_relationship_threshold: float = 0.62
    propagation_create_memories: bool = True
    propagation_memory_salience_multiplier: float = 0.45


class RoleConfig(BaseModel):
    role_influence_enabled: bool = True
    role_influence_min_score: float = 0.45
    role_influence_max_adjustment: float = 0.08
    role_influence_relevant_actions_only: bool = True


class NormConfig(BaseModel):
    norm_influence_enabled: bool = False
    norm_influence_strength: float = 0.02
    norm_min_agents_for_stable: int = 3
    norm_min_households_for_stable: int = 2
    norm_repeated_actor_diminishing_factor: float = 0.35
    norm_repeated_household_diminishing_factor: float = 0.65
    norm_stability_support_threshold: float = 1.2
    norm_decline_threshold: float = 0.35


class SimulationConfig(BaseModel):
    population_size: int = 50
    settlement: SettlementConfig
    starting_resources: StartingResourcesConfig = Field(default_factory=StartingResourcesConfig)
    household: HouseholdConfig = Field(default_factory=HouseholdConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    upkeep: UpkeepConfig = Field(default_factory=UpkeepConfig)
    propagation: PropagationConfig = Field(default_factory=PropagationConfig)
    roles: RoleConfig = Field(default_factory=RoleConfig)
    norms: NormConfig = Field(default_factory=NormConfig)


class ThreadedEarthConfig(BaseModel):
    simulation: SimulationConfig
    archetypes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ThreadedEarthConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ThreadedEarthConfig.model_validate(data)
