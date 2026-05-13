from __future__ import annotations

import random
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from threaded_earth.cognition import DecisionTrace, choose_action
from threaded_earth.config import ThreadedEarthConfig
from threaded_earth.events import record_event
from threaded_earth.generation import create_initial_state
from threaded_earth.goals import update_agent_goals
from threaded_earth.memory import retrieve_relevant_memories
from threaded_earth.metrics import write_metrics
from threaded_earth.models import Agent, Decision, Household, Memory, Relationship, Resource, Run
from threaded_earth.paths import ensure_artifact_dirs
from threaded_earth.snapshots import write_snapshot


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
        write_snapshot(session, run_id, tick)
        session.commit()
    run.status = "complete"
    run.updated_at = datetime.now(timezone.utc)
    session.commit()


def _simulate_tick(session: Session, run_id: str, tick: int, rng: random.Random) -> None:
    agents = session.query(Agent).filter(Agent.run_id == run_id, Agent.status == "active").order_by(Agent.neutral_id).all()
    agents_by_id = {agent.neutral_id: agent for agent in agents}
    households = {household.household_id: household for household in session.query(Household).filter(Household.run_id == run_id)}
    households_by_agent = {agent.neutral_id: households[agent.household_id] for agent in agents}
    for agent in agents:
        household = households[agent.household_id]
        relationships = (
            session.query(Relationship)
            .filter(Relationship.run_id == run_id, Relationship.source_agent == agent.neutral_id)
            .all()
        )
        retrieved_memories = retrieve_relevant_memories(session, run_id, agent.neutral_id, tick, relationships)
        active_goals = update_agent_goals(session, run_id, agent, household, relationships, retrieved_memories, tick)
        trace = choose_action(agent, household, relationships, rng, tick, retrieved_memories, active_goals, households_by_agent)
        _record_decision(session, run_id, agent, tick, trace)
        _apply_action(session, run_id, tick, rng, agent, household, relationships, trace, agents_by_id)
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
                f"retrieved_memory_ids={trace.retrieved_memory_ids}",
                f"memory_influence_summary={trace.memory_influence_summary}",
                f"memory_score_adjustments={trace.memory_score_adjustments}",
                f"active_goal_ids={trace.active_goal_ids}",
                f"goal_influence_summary={trace.goal_influence_summary}",
                f"goal_score_adjustments={trace.goal_score_adjustments}",
                f"selected_target_agent_id={trace.selected_target_agent_id}",
                f"target_selection_reasons={trace.target_selection_reasons}",
            ],
            retrieved_memory_ids=trace.retrieved_memory_ids,
            memory_influence_summary=trace.memory_influence_summary,
            memory_score_adjustments=trace.memory_score_adjustments,
            active_goal_ids=trace.active_goal_ids,
            goal_influence_summary=trace.goal_influence_summary,
            goal_score_adjustments=trace.goal_score_adjustments,
            selected_target_agent_id=trace.selected_target_agent_id,
            target_selection_candidates=trace.target_selection_candidates,
            target_selection_scores=trace.target_selection_scores,
            target_selection_reasons=trace.target_selection_reasons,
            target_memory_factors=trace.target_memory_factors,
            target_goal_factors=trace.target_goal_factors,
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
    agents_by_id: dict[str, Agent],
) -> None:
    action = trace.selected_action["action"]
    target_rel = _relationship_for_target(relationships, trace.selected_target_agent_id)
    target_name = _target_name(agents_by_id, trace.selected_target_agent_id)
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
            trace.selected_target_agent_id,
            _social_payload(trace, {"trust": 0.04, "affinity": 0.03}),
            f"{agent.display_name} cooperated with {target_name} on shared settlement work.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.58, event.summary)
        _remember_target(session, run_id, tick, trace.selected_target_agent_id, event.event_id, 0.56, event.summary)
    elif action == "share_food":
        _add_resource(session, run_id, household.household_id, "grain", -1.0)
        if target_rel is not None:
            target_rel.reputation = min(1.0, target_rel.reputation + 0.05)
            target_rel.trust = min(1.0, target_rel.trust + 0.03)
        event = record_event(
            session,
            run_id,
            tick,
            "resource_exchange",
            agent.neutral_id,
            trace.selected_target_agent_id,
            _social_payload(trace, {"trust": 0.03, "reputation": 0.05}, {"resource_delta": -1.0, "resource": "grain"}),
            f"{agent.display_name} shared stored grain with {target_name}.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.64, event.summary)
        _remember_target(session, run_id, tick, trace.selected_target_agent_id, event.event_id, 0.62, event.summary)
    elif action == "repair_relationship":
        if target_rel is not None:
            target_rel.trust = min(1.0, target_rel.trust + 0.05)
            target_rel.affinity = min(1.0, target_rel.affinity + 0.04)
            target_rel.reputation = min(1.0, target_rel.reputation + 0.02)
        agent.needs["belonging"] = min(100, agent.needs["belonging"] + 6)
        event = record_event(
            session,
            run_id,
            tick,
            "repair",
            agent.neutral_id,
            trace.selected_target_agent_id,
            _social_payload(trace, {"trust": 0.05, "affinity": 0.04, "reputation": 0.02}),
            f"{agent.display_name} tried to repair a strained tie with {target_name}.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.66, event.summary)
        _remember_target(session, run_id, tick, trace.selected_target_agent_id, event.event_id, 0.62, event.summary)
    elif action == "avoid_conflict":
        agent.needs["belonging"] = max(0, agent.needs["belonging"] - 2)
        event = record_event(
            session,
            run_id,
            tick,
            "avoidance",
            agent.neutral_id,
            trace.selected_target_agent_id,
            _social_payload(trace, {}, {"action": action}),
            f"{agent.display_name} avoided direct contact with {target_name}.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.57, event.summary)
        _remember_target(session, run_id, tick, trace.selected_target_agent_id, event.event_id, 0.55, event.summary)
    elif action == "conflict_over_food":
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
            trace.selected_target_agent_id,
            _social_payload(trace, {"trust": -0.06, "affinity": -0.05, "reputation": -0.03}),
            f"{agent.display_name} disputed access to scarce food with {target_name}.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.78, event.summary)
        _remember_target(session, run_id, tick, trace.selected_target_agent_id, event.event_id, 0.74, event.summary)
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


def _relationship_for_target(relationships: list[Relationship], target_agent_id: str | None) -> Relationship | None:
    if target_agent_id is None:
        return None
    for relationship in relationships:
        if relationship.target_agent == target_agent_id:
            return relationship
    return None


def _target_name(agents_by_id: dict[str, Agent], target_agent_id: str | None) -> str:
    if target_agent_id is None:
        return "no selected target"
    target = agents_by_id.get(target_agent_id)
    return target.display_name if target else target_agent_id


def _social_payload(
    trace: DecisionTrace,
    relationship_delta: dict[str, float],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "relationship_delta": relationship_delta,
        "active_needs": trace.active_needs,
        "selected_target_agent_id": trace.selected_target_agent_id,
        "target_selection_reasons": trace.target_selection_reasons,
        "target_selection_scores": trace.target_selection_scores,
        "target_memory_factors": trace.target_memory_factors,
        "target_goal_factors": trace.target_goal_factors,
    }
    if extra:
        payload.update(extra)
    return payload


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


def _remember_target(
    session: Session,
    run_id: str,
    tick: int,
    target_agent_id: str | None,
    event_id: str,
    salience: float,
    summary: str,
) -> None:
    if target_agent_id is None:
        return
    _remember(session, run_id, tick, target_agent_id, event_id, salience, summary)
