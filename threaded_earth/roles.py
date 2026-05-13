from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.models import Agent, Decision, Event, Relationship, RoleSignal


ROLE_THRESHOLD = 0.45
ROLE_NAMES = {
    "provider",
    "helper",
    "mediator",
    "craft_worker",
    "trader",
    "conflict_prone",
    "isolated",
    "trusted_neighbor",
}


def initialize_role_biases(session: Session, run_id: str, agents: list[Agent], tick: int = 0) -> None:
    for agent in agents:
        for role_name, amount in _archetype_biases(agent.archetype).items():
            update_role_signal(session, run_id, agent.neutral_id, role_name, amount, tick, f"archetype bias: {agent.archetype}")


def get_active_role_signals(session: Session, run_id: str, agent_id: str, min_score: float = 0.05) -> list[RoleSignal]:
    return (
        session.query(RoleSignal)
        .filter(RoleSignal.run_id == run_id, RoleSignal.agent_id == agent_id, RoleSignal.score >= min_score)
        .order_by(RoleSignal.score.desc(), RoleSignal.role_name)
        .all()
    )


def role_signals_by_agent(session: Session, run_id: str, min_score: float = 0.05) -> dict[str, list[RoleSignal]]:
    roles = (
        session.query(RoleSignal)
        .filter(RoleSignal.run_id == run_id, RoleSignal.score >= min_score)
        .order_by(RoleSignal.agent_id, RoleSignal.score.desc(), RoleSignal.role_name)
        .all()
    )
    grouped: dict[str, list[RoleSignal]] = {}
    for role in roles:
        grouped.setdefault(role.agent_id, []).append(role)
    return grouped


def update_role_signal(
    session: Session,
    run_id: str,
    agent_id: str,
    role_name: str,
    amount: float,
    tick: int,
    evidence_summary: str,
) -> RoleSignal:
    role = (
        session.query(RoleSignal)
        .filter(RoleSignal.run_id == run_id, RoleSignal.agent_id == agent_id, RoleSignal.role_name == role_name)
        .first()
    )
    if role is None:
        count = session.query(RoleSignal).filter(RoleSignal.run_id == run_id).count()
        role = RoleSignal(
            role_signal_id=f"{run_id}-role-{count + 1:06d}",
            run_id=run_id,
            agent_id=agent_id,
            role_name=role_name,
            score=0.0,
            evidence_count=0,
            created_tick=tick,
            updated_tick=tick,
            evidence_summary=evidence_summary,
        )
        session.add(role)
    role.score = round(min(1.0, max(0.0, role.score + amount)), 3)
    role.evidence_count += 1
    role.updated_tick = tick
    role.evidence_summary = evidence_summary
    session.flush()
    return role


def record_action_role_evidence(session: Session, run_id: str, agent: Agent, tick: int, action: str, event: Event) -> None:
    if action == "seek_food":
        update_role_signal(session, run_id, agent.neutral_id, "provider", 0.09, tick, f"resource gain: {event.event_id}")
        if agent.archetype in {"craft worker", "builder"}:
            update_role_signal(session, run_id, agent.neutral_id, "craft_worker", 0.08, tick, f"materials work: {event.event_id}")
    elif action == "cooperate":
        update_role_signal(session, run_id, agent.neutral_id, "helper", 0.09, tick, f"cooperation: {event.event_id}")
    elif action == "share_food":
        update_role_signal(session, run_id, agent.neutral_id, "trader", 0.09, tick, f"resource exchange: {event.event_id}")
        update_role_signal(session, run_id, agent.neutral_id, "helper", 0.03, tick, f"shared resources: {event.event_id}")
    elif action == "repair_relationship":
        update_role_signal(session, run_id, agent.neutral_id, "mediator", 0.1, tick, f"repair action: {event.event_id}")
    elif action == "conflict_over_food":
        update_role_signal(session, run_id, agent.neutral_id, "conflict_prone", 0.12, tick, f"conflict: {event.event_id}")
    elif action == "avoid_conflict":
        update_role_signal(session, run_id, agent.neutral_id, "isolated", 0.04, tick, f"avoidance: {event.event_id}")


def update_tick_role_signals(session: Session, run_id: str, agents: list[Agent], tick: int) -> None:
    decisions = session.query(Decision).filter(Decision.run_id == run_id, Decision.tick == tick).all()
    social_agents = {
        decision.agent_id
        for decision in decisions
        if decision.selected_action.get("action") in {"cooperate", "share_food", "repair_relationship", "avoid_conflict", "conflict_over_food"}
    }
    for decision in decisions:
        if decision.selected_target_agent_id:
            social_agents.add(decision.selected_target_agent_id)

    for agent in agents:
        relationships = (
            session.query(Relationship)
            .filter(Relationship.run_id == run_id, Relationship.source_agent == agent.neutral_id)
            .all()
        )
        if relationships:
            avg_trust = sum(relationship.trust for relationship in relationships) / len(relationships)
            avg_reputation = sum(relationship.reputation for relationship in relationships) / len(relationships)
            if avg_trust >= 0.64 and avg_reputation >= 0.56:
                update_role_signal(
                    session,
                    run_id,
                    agent.neutral_id,
                    "trusted_neighbor",
                    0.035,
                    tick,
                    f"average trust={avg_trust:.2f}, reputation={avg_reputation:.2f}",
                )
        if agent.neutral_id not in social_agents:
            update_role_signal(session, run_id, agent.neutral_id, "isolated", 0.025, tick, "low interaction frequency")


def role_adjustments(roles: list[RoleSignal]) -> dict[str, float]:
    adjustments = {
        "seek_food": 0.0,
        "cooperate": 0.0,
        "share_food": 0.0,
        "repair_relationship": 0.0,
        "avoid_conflict": 0.0,
        "conflict_over_food": 0.0,
    }
    for role in roles:
        weight = min(0.06, role.score * 0.08)
        if role.role_name == "provider":
            adjustments["seek_food"] += weight
            adjustments["cooperate"] += weight * 0.3
        elif role.role_name == "helper":
            adjustments["cooperate"] += weight
            adjustments["share_food"] += weight * 0.25
        elif role.role_name == "mediator":
            adjustments["repair_relationship"] += weight
            adjustments["cooperate"] += weight * 0.2
        elif role.role_name == "trader":
            adjustments["share_food"] += weight
        elif role.role_name == "conflict_prone":
            adjustments["conflict_over_food"] += min(0.025, weight * 0.6)
            adjustments["avoid_conflict"] += min(0.02, weight * 0.35)
        elif role.role_name == "isolated":
            adjustments["avoid_conflict"] += weight * 0.6
            adjustments["seek_food"] += weight * 0.25
            adjustments["cooperate"] -= min(0.02, weight * 0.3)
        elif role.role_name == "trusted_neighbor":
            adjustments["cooperate"] += weight * 0.35
            adjustments["share_food"] += weight * 0.3
        elif role.role_name == "craft_worker":
            adjustments["seek_food"] += weight * 0.25
            adjustments["cooperate"] += weight * 0.25
    return {action: round(value, 4) for action, value in adjustments.items() if abs(value) >= 0.001}


def summarize_role_influence(roles: list[RoleSignal], adjustments: dict[str, float]) -> str:
    if not roles:
        return "No role signals applied."
    role_text = ", ".join(f"{role.role_name}={role.score:.2f}" for role in roles[:4])
    if not adjustments:
        return f"Role signals present but below scoring threshold: {role_text}."
    adjustment_text = ", ".join(f"{action}={value:+.3f}" for action, value in sorted(adjustments.items()))
    return f"Roles {role_text} adjusted actions: {adjustment_text}."


def role_stats(session: Session, run_id: str) -> dict[str, Any]:
    roles = session.query(RoleSignal).filter(RoleSignal.run_id == run_id).order_by(RoleSignal.score.desc(), RoleSignal.role_name).all()
    counts: dict[str, int] = {}
    top = []
    strongest_by_agent: dict[str, dict[str, Any]] = {}
    for role in roles:
        if role.score >= ROLE_THRESHOLD:
            counts[role.role_name] = counts.get(role.role_name, 0) + 1
        if len(top) < 8:
            top.append(_role_payload(role))
        current = strongest_by_agent.get(role.agent_id)
        if current is None or role.score > current["score"]:
            strongest_by_agent[role.agent_id] = _role_payload(role)
    return {
        "role_counts_above_threshold": dict(sorted(counts.items())),
        "top_role_signals": top,
        "agents_with_strongest_role_scores": sorted(
            strongest_by_agent.values(), key=lambda item: (-item["score"], item["agent_id"], item["role_name"])
        )[:8],
    }


def _role_payload(role: RoleSignal) -> dict[str, Any]:
    return {
        "agent_id": role.agent_id,
        "role_name": role.role_name,
        "score": round(role.score, 3),
        "evidence_count": role.evidence_count,
        "updated_tick": role.updated_tick,
        "evidence_summary": role.evidence_summary,
    }


def _archetype_biases(archetype: str) -> dict[str, float]:
    if archetype in {"cultivator", "gatherer", "fisher", "hunter"}:
        return {"provider": 0.08}
    if archetype in {"healer", "elder", "storyteller"}:
        return {"helper": 0.06, "mediator": 0.04}
    if archetype == "trader":
        return {"trader": 0.08}
    if archetype in {"craft worker", "builder"}:
        return {"craft_worker": 0.08, "provider": 0.03}
    return {}
