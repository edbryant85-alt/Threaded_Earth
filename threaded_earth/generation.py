from __future__ import annotations

import random
from dataclasses import dataclass

from sqlalchemy.orm import Session

from threaded_earth.config import ThreadedEarthConfig
from threaded_earth.models import Agent, Household, Location, Relationship, Resource


NAME_ROOTS = [
    "Ala",
    "Bira",
    "Cala",
    "Dema",
    "Ena",
    "Faro",
    "Hala",
    "Ilo",
    "Jora",
    "Kavi",
    "Luma",
    "Mira",
    "Naro",
    "Ona",
    "Pela",
    "Rani",
    "Sela",
    "Tavi",
    "Una",
    "Velo",
    "Wira",
    "Yani",
    "Zora",
]

TRAIT_NAMES = ["patient", "bold", "practical", "curious", "generous", "guarded", "steady"]
AGE_BANDS = ["youth", "adult", "adult", "adult", "elder"]
KINSHIP_TYPES = ["extended", "sibling-led", "parent-child", "mixed kin"]


@dataclass(frozen=True)
class PopulationResult:
    households: list[Household]
    agents: list[Agent]


def create_initial_state(session: Session, run_id: str, seed: int, config: ThreadedEarthConfig) -> PopulationResult:
    rng = random.Random(seed)
    settlement = config.simulation.settlement
    session.add(
        Location(
            location_id=settlement.location_id,
            run_id=run_id,
            name=settlement.name,
            symbolic_description=settlement.symbolic_description,
        )
    )

    households = _build_households(run_id, rng, config)
    agents = _build_agents(run_id, rng, config, households)
    _assign_household_members(households, agents)

    session.add_all(households)
    session.add_all(agents)
    _create_household_resources(session, run_id, households, config)
    _create_relationships(session, run_id, rng, agents)
    return PopulationResult(households=households, agents=agents)


def _build_households(run_id: str, rng: random.Random, config: ThreadedEarthConfig) -> list[Household]:
    population = config.simulation.population_size
    min_size = config.simulation.household.min_size
    max_size = config.simulation.household.max_size
    remaining = population
    households: list[Household] = []
    index = 1
    while remaining > 0:
        size = min(max_size, max(min_size, rng.randint(min_size, max_size)))
        if remaining - size < min_size and remaining - size != 0:
            size = remaining
        size = min(size, remaining)
        household_id = f"hh-{index:03d}"
        households.append(
            Household(
                household_id=household_id,
                run_id=run_id,
                household_name=f"{rng.choice(NAME_ROOTS)} Hearth",
                kinship_type=rng.choice(KINSHIP_TYPES),
                members=[],
                stored_resources={},
            )
        )
        remaining -= size
        index += 1
    return households


def _build_agents(
    run_id: str,
    rng: random.Random,
    config: ThreadedEarthConfig,
    households: list[Household],
) -> list[Agent]:
    agents: list[Agent] = []
    household_slots: list[str] = []
    base = config.simulation.population_size // len(households)
    extra = config.simulation.population_size % len(households)
    for index, household in enumerate(households):
        household_slots.extend([household.household_id] * (base + (1 if index < extra else 0)))

    archetypes = _balanced_archetypes(config.archetypes, config.simulation.population_size)
    rng.shuffle(archetypes)
    for index in range(config.simulation.population_size):
        neutral_id = f"agent-{index + 1:03d}"
        agent = Agent(
            neutral_id=neutral_id,
            run_id=run_id,
            display_name=_unique_name(rng, index),
            archetype=archetypes[index],
            age_band=rng.choice(AGE_BANDS),
            household_id=household_slots[index],
            traits=_traits(rng),
            needs={"food": rng.randint(28, 48), "rest": rng.randint(18, 35), "belonging": rng.randint(18, 42)},
            status="active",
        )
        agents.append(agent)
    return agents


def _balanced_archetypes(archetypes: list[str], population_size: int) -> list[str]:
    repeats = (population_size // len(archetypes)) + 1
    return (archetypes * repeats)[:population_size]


def _unique_name(rng: random.Random, index: int) -> str:
    root = rng.choice(NAME_ROOTS)
    suffix = ["na", "ri", "lo", "mi", "ta", "su"][index % 6]
    return f"{root}{suffix}"


def _traits(rng: random.Random) -> dict[str, float]:
    chosen = rng.sample(TRAIT_NAMES, 3)
    return {trait: round(rng.uniform(0.25, 0.85), 2) for trait in chosen}


def _assign_household_members(households: list[Household], agents: list[Agent]) -> None:
    by_household = {household.household_id: household for household in households}
    for agent in agents:
        by_household[agent.household_id].members.append(agent.neutral_id)


def _create_household_resources(
    session: Session,
    run_id: str,
    households: list[Household],
    config: ThreadedEarthConfig,
) -> None:
    resources = config.simulation.starting_resources
    for household in households:
        stored = {
            "grain": float(resources.grain_per_household),
            "fish": float(resources.fish_per_household),
            "reeds": float(resources.reeds_per_household),
            "tools": float(resources.tools_per_household),
        }
        household.stored_resources = stored
        for resource_type, quantity in stored.items():
            session.add(
                Resource(
                    run_id=run_id,
                    resource_type=resource_type,
                    owner_scope="household",
                    owner_id=household.household_id,
                    quantity=quantity,
                )
            )


def _create_relationships(session: Session, run_id: str, rng: random.Random, agents: list[Agent]) -> None:
    by_household: dict[str, list[Agent]] = {}
    for agent in agents:
        by_household.setdefault(agent.household_id, []).append(agent)

    for members in by_household.values():
        for source in members:
            for target in members:
                if source.neutral_id == target.neutral_id:
                    continue
                session.add(
                    Relationship(
                        run_id=run_id,
                        source_agent=source.neutral_id,
                        target_agent=target.neutral_id,
                        affinity=round(rng.uniform(0.45, 0.78), 3),
                        trust=round(rng.uniform(0.5, 0.82), 3),
                        reputation=round(rng.uniform(0.45, 0.7), 3),
                        kinship_relation="household kin",
                    )
                )

    for source in agents:
        candidates = [agent for agent in agents if agent.household_id != source.household_id]
        for target in rng.sample(candidates, k=min(3, len(candidates))):
            session.add(
                Relationship(
                    run_id=run_id,
                    source_agent=source.neutral_id,
                    target_agent=target.neutral_id,
                    affinity=round(rng.uniform(0.22, 0.62), 3),
                    trust=round(rng.uniform(0.2, 0.58), 3),
                    reputation=round(rng.uniform(0.35, 0.62), 3),
                    kinship_relation="settlement tie",
                )
            )
