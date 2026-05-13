from __future__ import annotations

import random
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from threaded_earth.cognition import DecisionTrace, choose_action
from threaded_earth.config import ThreadedEarthConfig
from threaded_earth.events import record_event
from threaded_earth.generation import create_initial_state
from threaded_earth.metrics import write_metrics
from threaded_earth.models import Agent, Decision, Household, Memory, Relationship, Resource, Run
from threaded_earth.paths import ensure_artifact_dirs


def make_run_id(seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"run-{stamp}-seed-{seed}"


def initialize_run(session: Session, run_id: str, seed: int, config: ThreadedEarthConfig) -> None:
    ensure_artifact_dirs(run_id)
    session.add(Run(run_id=run_id, seed=seed, config=config.as_dict(), status="initialized"))
    create_initial_state(session, run_id, seed, config)
    record_event(
        session,
        run_id,
        0,
        "run_initialized",
        None,
        None,
        {"population_size": config.simulation.population_size},
        "Settlement initialized in the Threaded River Valley.",
    )
    session.commit()


def run_simulation(session: Session, run_id: str, days: int, seed: int, config: ThreadedEarthConfig) -> None:
    rng = random.Random(seed + 991)
    run = session.get(Run, run_id)
    if run is None:
        initialize_run(session, run_id, seed, config)
        run = session.get(Run, run_id)
    if run is None:
        raise RuntimeError(f"Run {run_id} could not be initialized")

    run.status = "running"
    for tick in range(1, days + 1):
        _simulate_tick(session, run_id, tick, rng)
        write_metrics(session, run_id)
        session.commit()
    run.status = "complete"
    run.updated_at = datetime.now(timezone.utc)
    session.commit()


def _simulate_tick(session: Session, run_id: str, tick: int, rng: random.Random) -> None:
    agents = session.query(Agent).filter(Agent.run_id == run_id, Agent.status == "active").order_by(Agent.neutral_id).all()
    households = {household.household_id: household for household in session.query(Household).filter(Household.run_id == run_id)}
    for agent in agents:
        household = households[agent.household_id]
        relationships = (
            session.query(Relationship)
            .filter(Relationship.run_id == run_id, Relationship.source_agent == agent.neutral_id)
            .all()
        )
        trace = choose_action(agent, household, relationships, rng, tick)
        _record_decision(session, run_id, agent, tick, trace)
        _apply_action(session, run_id, tick, rng, agent, household, relationships, trace)
        _decay_needs(agent)


def _record_decision(session: Session, run_id: str, agent: Agent, tick: int, trace: DecisionTrace) -> None:
    session.add(
        Decision(
            run_id=run_id,
            agent_id=agent.neutral_id,
            tick=tick,
            candidate_actions=trace.candidate_actions,
            selected_action=trace.selected_action,
            reasons=trace.reasons
            + [
                f"active_needs={trace.active_needs}",
                f"relationship_modifiers={trace.relationship_modifiers}",
                f"memories_consulted={trace.memories_consulted}",
            ],
            confidence=trace.confidence,
            uncertainty_notes=trace.uncertainty_notes,
        )
    )


def _apply_action(
    session: Session,
    run_id: str,
    tick: int,
    rng: random.Random,
    agent: Agent,
    household: Household,
    relationships: list[Relationship],
    trace: DecisionTrace,
) -> None:
    action = trace.selected_action["action"]
    if action == "seek_food":
        amount = _food_yield(agent.archetype, rng)
        _add_resource(session, run_id, household.household_id, "grain" if agent.archetype != "fisher" else "fish", amount)
        agent.needs["food"] = min(100, agent.needs["food"] + 8)
        event = record_event(
            session,
            run_id,
            tick,
            "resource_change",
            agent.neutral_id,
            household.household_id,
            {"resource_delta": amount, "action": action, "active_needs": trace.active_needs},
            f"{agent.display_name} gathered food for {household.household_name}.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.45, event.summary)
    elif action == "cooperate":
        target_rel = _pick_relationship(relationships, rng)
        if target_rel is not None:
            target_rel.trust = min(1.0, target_rel.trust + 0.04)
            target_rel.affinity = min(1.0, target_rel.affinity + 0.03)
        agent.needs["belonging"] = min(100, agent.needs["belonging"] + 9)
        event = record_event(
            session,
            run_id,
            tick,
            "cooperation",
            agent.neutral_id,
            target_rel.target_agent if target_rel else None,
            {"relationship_delta": {"trust": 0.04, "affinity": 0.03}, "active_needs": trace.active_needs},
            f"{agent.display_name} cooperated on shared settlement work.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.58, event.summary)
    elif action == "share_food":
        _add_resource(session, run_id, household.household_id, "grain", -1.0)
        target_rel = _pick_relationship(relationships, rng)
        if target_rel is not None:
            target_rel.reputation = min(1.0, target_rel.reputation + 0.05)
            target_rel.trust = min(1.0, target_rel.trust + 0.03)
        event = record_event(
            session,
            run_id,
            tick,
            "resource_exchange",
            agent.neutral_id,
            target_rel.target_agent if target_rel else None,
            {"resource_delta": -1.0, "resource": "grain", "active_needs": trace.active_needs},
            f"{agent.display_name} shared stored grain with another tie.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.64, event.summary)
    elif action == "conflict_over_food":
        target_rel = _pick_relationship(relationships, rng)
        if target_rel is not None:
            target_rel.trust = max(0.0, target_rel.trust - 0.06)
            target_rel.affinity = max(0.0, target_rel.affinity - 0.05)
            target_rel.reputation = max(0.0, target_rel.reputation - 0.03)
        agent.needs["belonging"] = max(0, agent.needs["belonging"] - 5)
        event = record_event(
            session,
            run_id,
            tick,
            "conflict",
            agent.neutral_id,
            target_rel.target_agent if target_rel else None,
            {"relationship_delta": {"trust": -0.06, "affinity": -0.05}, "active_needs": trace.active_needs},
            f"{agent.display_name} disputed access to scarce food.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.78, event.summary)
    else:
        agent.needs["rest"] = min(100, agent.needs["rest"] + 14)
        event = record_event(
            session,
            run_id,
            tick,
            "rest",
            agent.neutral_id,
            None,
            {"active_needs": trace.active_needs},
            f"{agent.display_name} rested and reduced immediate strain.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.35, event.summary)


def _food_yield(archetype: str, rng: random.Random) -> float:
    base = {"cultivator": 2.4, "gatherer": 1.8, "fisher": 2.0, "hunter": 1.7}.get(archetype, 1.0)
    return round(base + rng.uniform(0.0, 1.2), 2)


def _pick_relationship(relationships: list[Relationship], rng: random.Random) -> Relationship | None:
    if not relationships:
        return None
    return rng.choice(relationships)


def _add_resource(session: Session, run_id: str, household_id: str, resource_type: str, amount: float) -> None:
    resource = (
        session.query(Resource)
        .filter(
            Resource.run_id == run_id,
            Resource.owner_scope == "household",
            Resource.owner_id == household_id,
            Resource.resource_type == resource_type,
        )
        .one()
    )
    resource.quantity = max(0.0, round(resource.quantity + amount, 2))
    household = session.get(Household, household_id)
    if household is not None:
        stored = dict(household.stored_resources)
        stored[resource_type] = resource.quantity
        household.stored_resources = stored


def _decay_needs(agent: Agent) -> None:
    needs = dict(agent.needs)
    needs["food"] = max(0, needs["food"] - 4)
    needs["rest"] = max(0, needs["rest"] - 3)
    needs["belonging"] = max(0, needs["belonging"] - 2)
    agent.needs = needs


def _remember(
    session: Session,
    run_id: str,
    tick: int,
    agent_id: str,
    event_id: str,
    salience: float,
    summary: str,
) -> None:
    if salience < 0.55:
        return
    count = session.query(Memory).filter(Memory.run_id == run_id).count()
    session.add(
        Memory(
            memory_id=f"{run_id}-mem-{count + 1:06d}",
            run_id=run_id,
            agent_id=agent_id,
            event_id=event_id,
            salience=salience,
            summary=summary,
            created_tick=tick,
        )
    )
