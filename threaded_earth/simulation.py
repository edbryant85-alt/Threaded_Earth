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
from threaded_earth.models import Agent, Decision, Household, Memory, Relationship, Run
from threaded_earth.norms import update_norm_candidates_for_event
from threaded_earth.paths import ensure_artifact_dirs
from threaded_earth.propagation import propagate_social_event
from threaded_earth.resources import add_household_resource, consume_household_food, household_food, transfer_household_resource
from threaded_earth.roles import (
    get_active_role_signals,
    initialize_role_biases,
    record_action_role_evidence,
    role_signals_by_agent,
    update_tick_role_signals,
)
from threaded_earth.snapshots import write_snapshot


def make_run_id(seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"run-{stamp}-seed-{seed}"


def initialize_run(session: Session, run_id: str, seed: int, config: ThreadedEarthConfig) -> None:
    ensure_artifact_dirs(run_id)
    session.add(Run(run_id=run_id, seed=seed, config=config.as_dict(), status="initialized"))
    create_initial_state(session, run_id, seed, config)
    agents = session.query(Agent).filter(Agent.run_id == run_id).order_by(Agent.neutral_id).all()
    initialize_role_biases(session, run_id, agents)
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
        _simulate_tick(session, run_id, tick, rng, config)
        write_metrics(session, run_id)
        write_snapshot(session, run_id, tick)
        session.commit()
    run.status = "complete"
    run.updated_at = datetime.now(timezone.utc)
    session.commit()


def _simulate_tick(session: Session, run_id: str, tick: int, rng: random.Random, config: ThreadedEarthConfig) -> None:
    agents = session.query(Agent).filter(Agent.run_id == run_id, Agent.status == "active").order_by(Agent.neutral_id).all()
    agents_by_id = {agent.neutral_id: agent for agent in agents}
    households = {household.household_id: household for household in session.query(Household).filter(Household.run_id == run_id)}
    households_by_agent = {agent.neutral_id: households[agent.household_id] for agent in agents}
    _apply_household_upkeep(session, run_id, tick, households, agents, agents_by_id, households_by_agent, config)
    roles_by_agent = role_signals_by_agent(session, run_id)
    for agent in agents:
        household = households[agent.household_id]
        relationships = (
            session.query(Relationship)
            .filter(Relationship.run_id == run_id, Relationship.source_agent == agent.neutral_id)
            .all()
        )
        retrieved_memories = retrieve_relevant_memories(session, run_id, agent.neutral_id, tick, relationships)
        active_goals = update_agent_goals(session, run_id, agent, household, relationships, retrieved_memories, tick)
        active_roles = roles_by_agent.get(agent.neutral_id) or get_active_role_signals(session, run_id, agent.neutral_id)
        trace = choose_action(
            agent,
            household,
            relationships,
            rng,
            tick,
            retrieved_memories,
            active_goals,
            households_by_agent,
            active_roles,
            roles_by_agent,
            config.simulation.roles,
        )
        _record_decision(session, run_id, agent, tick, trace)
        _apply_action(session, run_id, tick, rng, agent, household, relationships, trace, agents_by_id, households_by_agent, config)
        _decay_needs(agent)
    update_tick_role_signals(session, run_id, agents, tick)


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
                f"role_influence_summary={trace.role_influence_summary}",
                f"role_signals_seen={trace.role_signals_seen}",
                f"role_signals_applied={trace.role_signals_applied}",
                f"role_score_adjustments={trace.role_score_adjustments}",
                f"role_adjustment_total={trace.role_adjustment_total}",
                f"role_adjustment_capped={trace.role_adjustment_capped}",
                f"selected_target_agent_id={trace.selected_target_agent_id}",
                f"target_selection_reasons={trace.target_selection_reasons}",
                f"target_aware_action_scores={trace.target_aware_action_scores}",
                f"best_target_by_action={trace.best_target_by_action}",
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
            target_aware_action_scores=trace.target_aware_action_scores,
            best_target_by_action=trace.best_target_by_action,
            target_aware_score_reasons=trace.target_aware_score_reasons,
            final_score_breakdown=trace.final_score_breakdown,
            role_influence_summary=trace.role_influence_summary,
            role_score_adjustments=trace.role_score_adjustments,
            role_signals_seen=trace.role_signals_seen,
            role_signals_applied=trace.role_signals_applied,
            role_adjustment_total=trace.role_adjustment_total,
            role_adjustment_capped=trace.role_adjustment_capped,
            confidence=trace.confidence,
            uncertainty_notes=trace.uncertainty_notes,
        )
    )


def _apply_household_upkeep(
    session: Session,
    run_id: str,
    tick: int,
    households: dict[str, Household],
    agents: list[Agent],
    agents_by_id: dict[str, Agent],
    households_by_agent: dict[str, Household],
    config: ThreadedEarthConfig,
) -> None:
    members_by_household: dict[str, list[Agent]] = {household_id: [] for household_id in households}
    for agent in agents:
        members_by_household.setdefault(agent.household_id, []).append(agent)
    upkeep = config.simulation.upkeep
    for household_id, household in sorted(households.items()):
        members = members_by_household.get(household_id, [])
        requested = len(members) * upkeep.daily_food_need_per_agent
        result = consume_household_food(session, run_id, household_id, requested)
        material_decay = 0.0
        if upkeep.material_decay_enabled and upkeep.material_decay_per_household > 0:
            material_decay = abs(add_household_resource(session, run_id, household_id, "reeds", -upkeep.material_decay_per_household))
        payload = {
            "household_id": household_id,
            "member_count": len(members),
            "daily_food_need_per_agent": upkeep.daily_food_need_per_agent,
            "food_requested": result["requested_food"],
            "food_consumed": result["consumed_food"],
            "shortage_amount": result["shortage_amount"],
            "consumed_by_type": result["consumed_by_type"],
            "material_decay": material_decay,
            "household_food_after": household_food(household),
        }
        event_type = "household_shortage" if result["shortage_amount"] > 0 else "household_upkeep"
        summary = (
            f"{household.household_name} consumed {result['consumed_food']} food for daily upkeep."
            if result["shortage_amount"] <= 0
            else f"{household.household_name} consumed {result['consumed_food']} food and faced a shortage of {result['shortage_amount']}."
        )
        event = record_event(session, run_id, tick, event_type, None, household_id, payload, summary)
        if result["shortage_amount"] > 0:
            shortage_per_member = result["shortage_amount"] / len(members) if members else 0.0
            for member in members:
                needs = dict(member.needs)
                needs["food"] = max(0, needs.get("food", 0) - round(2 + shortage_per_member * 4, 2))
                member.needs = needs
                if result["shortage_amount"] >= upkeep.food_shortage_memory_threshold:
                    _remember(session, run_id, tick, member.neutral_id, event.event_id, 0.7, summary)
            if result["shortage_amount"] >= upkeep.food_shortage_memory_threshold:
                propagate_social_event(
                    session,
                    run_id,
                    tick,
                    event,
                    agents_by_id,
                    households_by_agent,
                    config.simulation.propagation,
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
    households_by_agent: dict[str, Household],
    config: ThreadedEarthConfig,
) -> None:
    action = trace.selected_action["action"]
    target_rel = _relationship_for_target(relationships, trace.selected_target_agent_id)
    target_name = _target_name(agents_by_id, trace.selected_target_agent_id)
    if action == "seek_food":
        resource_type, amount = _resource_yield(agent.archetype, rng)
        actual_delta = add_household_resource(session, run_id, household.household_id, resource_type, amount)
        agent.needs["food"] = min(100, agent.needs["food"] + 8)
        event = record_event(
            session,
            run_id,
            tick,
            "resource_change",
            agent.neutral_id,
            household.household_id,
            {
                "resource_delta": actual_delta,
                "resource": resource_type,
                "resource_transfer": None,
                "action": action,
                "active_needs": trace.active_needs,
                "household_food_after": household_food(household),
            },
            f"{agent.display_name} gathered {resource_type} for {household.household_name}.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.45, event.summary)
        record_action_role_evidence(session, run_id, agent, tick, action, event)
        update_norm_candidates_for_event(session, run_id, event, agent, None, target_rel, tick, config.simulation.norms)
    elif action == "cooperate":
        target_household = _target_household(households_by_agent, trace.selected_target_agent_id)
        transfer = None
        if target_household is not None:
            helped_resource = "reeds" if agent.archetype in {"builder", "craft worker"} else "grain"
            helped_amount = 0.7 if helped_resource == "grain" else 0.45
            actual_delta = add_household_resource(session, run_id, target_household.household_id, helped_resource, helped_amount)
            transfer = {
                "resource_type": helped_resource,
                "requested_quantity": helped_amount,
                "transferred_quantity": actual_delta,
                "source_household_id": None,
                "target_household_id": target_household.household_id,
                "status": "created_by_cooperation",
                "reason": "cooperative work increased target household stores",
            }
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
            _social_payload(trace, {"trust": 0.04, "affinity": 0.03}, {"resource_transfer": transfer}),
            f"{agent.display_name} cooperated with {target_name} and improved household stores.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.58, event.summary)
        _remember_target(session, run_id, tick, trace.selected_target_agent_id, event.event_id, 0.56, event.summary)
        record_action_role_evidence(session, run_id, agent, tick, action, event)
        update_norm_candidates_for_event(
            session, run_id, event, agent, agents_by_id.get(trace.selected_target_agent_id), target_rel, tick, config.simulation.norms
        )
        propagate_social_event(session, run_id, tick, event, agents_by_id, households_by_agent, config.simulation.propagation)
    elif action == "share_food":
        target_household = _target_household(households_by_agent, trace.selected_target_agent_id)
        requested = 1.5 if household_food(household) >= 24 else 0.8
        transfer = transfer_household_resource(
            session,
            run_id,
            household.household_id,
            target_household.household_id if target_household else None,
            "grain",
            requested,
        )
        if target_rel is not None:
            if transfer["transferred_quantity"] > 0:
                target_rel.reputation = min(1.0, target_rel.reputation + 0.05)
                target_rel.trust = min(1.0, target_rel.trust + 0.03)
            else:
                target_rel.trust = max(0.0, target_rel.trust - 0.01)
        event = record_event(
            session,
            run_id,
            tick,
            "resource_exchange",
            agent.neutral_id,
            trace.selected_target_agent_id,
            _social_payload(
                trace,
                {"trust": 0.03, "reputation": 0.05} if transfer["transferred_quantity"] > 0 else {"trust": -0.01},
                {"resource_transfer": transfer},
            ),
            _share_summary(agent.display_name, target_name, transfer),
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.64, event.summary)
        _remember_target(session, run_id, tick, trace.selected_target_agent_id, event.event_id, 0.62, event.summary)
        record_action_role_evidence(session, run_id, agent, tick, action, event)
        update_norm_candidates_for_event(
            session, run_id, event, agent, agents_by_id.get(trace.selected_target_agent_id), target_rel, tick, config.simulation.norms
        )
        propagate_social_event(session, run_id, tick, event, agents_by_id, households_by_agent, config.simulation.propagation)
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
        record_action_role_evidence(session, run_id, agent, tick, action, event)
        update_norm_candidates_for_event(
            session, run_id, event, agent, agents_by_id.get(trace.selected_target_agent_id), target_rel, tick, config.simulation.norms
        )
        propagate_social_event(session, run_id, tick, event, agents_by_id, households_by_agent, config.simulation.propagation)
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
        record_action_role_evidence(session, run_id, agent, tick, action, event)
        update_norm_candidates_for_event(
            session, run_id, event, agent, agents_by_id.get(trace.selected_target_agent_id), target_rel, tick, config.simulation.norms
        )
    elif action == "conflict_over_food":
        target_household = _target_household(households_by_agent, trace.selected_target_agent_id)
        transfer = None
        if target_household is not None:
            before = float(target_household.stored_resources.get("grain", 0.0))
            damaged = abs(add_household_resource(session, run_id, target_household.household_id, "grain", -0.6))
            transfer = {
                "resource_type": "grain",
                "requested_quantity": 0.6,
                "transferred_quantity": damaged,
                "source_household_id": target_household.household_id,
                "target_household_id": None,
                "status": "damaged" if damaged > 0 else "failed",
                "reason": "conflict damaged or consumed target household stores",
                "source_quantity_before": round(before, 2),
                "source_quantity_after": round(float(target_household.stored_resources.get("grain", 0.0)), 2),
            }
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
            _social_payload(trace, {"trust": -0.06, "affinity": -0.05, "reputation": -0.03}, {"resource_transfer": transfer}),
            f"{agent.display_name} disputed access to scarce food with {target_name}, damaging household stores.",
        )
        _remember(session, run_id, tick, agent.neutral_id, event.event_id, 0.78, event.summary)
        _remember_target(session, run_id, tick, trace.selected_target_agent_id, event.event_id, 0.74, event.summary)
        record_action_role_evidence(session, run_id, agent, tick, action, event)
        update_norm_candidates_for_event(
            session, run_id, event, agent, agents_by_id.get(trace.selected_target_agent_id), target_rel, tick, config.simulation.norms
        )
        propagate_social_event(session, run_id, tick, event, agents_by_id, households_by_agent, config.simulation.propagation)
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


def _resource_yield(archetype: str, rng: random.Random) -> tuple[str, float]:
    if archetype == "fisher":
        return "fish", round(2.0 + rng.uniform(0.0, 1.2), 2)
    if archetype in {"builder", "craft worker"}:
        return "reeds", round(1.0 + rng.uniform(0.0, 0.8), 2)
    base = {"cultivator": 2.4, "gatherer": 1.8, "hunter": 1.7}.get(archetype, 1.0)
    return "grain", round(base + rng.uniform(0.0, 1.2), 2)


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


def _target_household(households_by_agent: dict[str, Household], target_agent_id: str | None) -> Household | None:
    if target_agent_id is None:
        return None
    return households_by_agent.get(target_agent_id)


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


def _share_summary(actor_name: str, target_name: str, transfer: dict[str, object]) -> str:
    amount = transfer["transferred_quantity"]
    if amount:
        return f"{actor_name} shared {amount} {transfer['resource_type']} with {target_name}."
    return f"{actor_name} tried to share {transfer['resource_type']} with {target_name}, but household stores were insufficient."


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
