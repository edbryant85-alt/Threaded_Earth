from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.memory import RetrievedMemory
from threaded_earth.models import Agent, Goal, Household, Relationship


GOAL_TYPES = {
    "secure_food",
    "maintain_household",
    "repair_relationship",
    "seek_cooperation",
    "avoid_conflict",
    "improve_reputation",
    "rest_or_recover",
}


def update_agent_goals(
    session: Session,
    run_id: str,
    agent: Agent,
    household: Household,
    relationships: list[Relationship],
    memories: list[RetrievedMemory],
    tick: int,
) -> list[Goal]:
    desired = _desired_goals(agent, household, relationships, memories)
    active = {
        goal.goal_type: goal
        for goal in session.query(Goal)
        .filter(Goal.run_id == run_id, Goal.agent_id == agent.neutral_id, Goal.status == "active")
        .all()
    }

    for goal_type, (priority, reason) in desired.items():
        if goal_type in active:
            goal = active[goal_type]
            goal.priority = round(min(1.0, max(goal.priority * 0.75, priority) + 0.04), 3)
            goal.updated_tick = tick
            goal.source_reason = reason
            goal.notes = "Updated from current needs, relationships, and memories."
        else:
            goal = Goal(
                goal_id=_next_goal_id(session, run_id),
                run_id=run_id,
                agent_id=agent.neutral_id,
                goal_type=goal_type,
                priority=round(priority, 3),
                status="active",
                created_tick=tick,
                updated_tick=tick,
                source_reason=reason,
                progress=0.0,
                notes="Created by symbolic goal update rules.",
            )
            session.add(goal)
            session.flush()
            active[goal_type] = goal

    for goal_type, goal in active.items():
        if goal_type in desired:
            continue
        goal.priority = round(max(0.0, goal.priority - 0.08), 3)
        goal.updated_tick = tick
        goal.progress = round(min(1.0, goal.progress + 0.12), 3)
        goal.notes = "Priority decayed because triggering condition was not present."
        if goal.priority < 0.18:
            goal.status = "satisfied"
            goal.notes = "Marked satisfied after priority decayed below threshold."

    return (
        session.query(Goal)
        .filter(Goal.run_id == run_id, Goal.agent_id == agent.neutral_id, Goal.status == "active")
        .order_by(Goal.priority.desc(), Goal.goal_id)
        .limit(4)
        .all()
    )


def goal_adjustments(goals: list[Goal]) -> dict[str, float]:
    adjustments = {
        "seek_food": 0.0,
        "cooperate": 0.0,
        "rest": 0.0,
        "share_food": 0.0,
        "conflict_over_food": 0.0,
        "avoid_conflict": 0.0,
        "repair_relationship": 0.0,
    }
    for goal in goals:
        weight = min(0.1, goal.priority * 0.08)
        if goal.goal_type == "secure_food":
            adjustments["seek_food"] += weight
            adjustments["share_food"] -= weight * 0.35
        elif goal.goal_type == "maintain_household":
            adjustments["seek_food"] += weight * 0.75
            adjustments["share_food"] += weight * 0.3
        elif goal.goal_type == "repair_relationship":
            adjustments["repair_relationship"] += weight
            adjustments["cooperate"] += weight
            adjustments["share_food"] += weight * 0.45
            adjustments["conflict_over_food"] -= weight * 0.75
        elif goal.goal_type == "seek_cooperation":
            adjustments["cooperate"] += weight
            adjustments["share_food"] += weight * 0.4
        elif goal.goal_type == "avoid_conflict":
            adjustments["avoid_conflict"] += weight
            adjustments["conflict_over_food"] -= weight
            adjustments["cooperate"] += weight * 0.25
        elif goal.goal_type == "improve_reputation":
            adjustments["share_food"] += weight
            adjustments["cooperate"] += weight * 0.5
        elif goal.goal_type == "rest_or_recover":
            adjustments["rest"] += weight
    return {key: round(value, 3) for key, value in adjustments.items() if abs(value) >= 0.001}


def summarize_goal_influence(goals: list[Goal], adjustments: dict[str, float]) -> str:
    if not goals:
        return "No active goals applied."
    goal_bits = ", ".join(f"{goal.goal_type}:{goal.priority:.2f}" for goal in goals)
    adjustment_bits = ", ".join(f"{action} {delta:+.3f}" for action, delta in sorted(adjustments.items()))
    return f"Active goals ({goal_bits}); score adjustments: {adjustment_bits or 'none'}."


def goal_stats(session: Session, run_id: str) -> dict[str, Any]:
    goals = session.query(Goal).filter(Goal.run_id == run_id).all()
    active = [goal for goal in goals if goal.status == "active"]
    by_type: dict[str, int] = {}
    for goal in active:
        by_type[goal.goal_type] = by_type.get(goal.goal_type, 0) + 1
    active_agents = len({goal.agent_id for goal in active})
    return {
        "total_active_goals": len(active),
        "goals_by_type": dict(sorted(by_type.items())),
        "average_active_goals_per_agent": round(len(active) / active_agents, 2) if active_agents else 0.0,
        "satisfied_count": sum(1 for goal in goals if goal.status == "satisfied"),
        "abandoned_count": sum(1 for goal in goals if goal.status == "abandoned"),
    }


def _desired_goals(
    agent: Agent,
    household: Household,
    relationships: list[Relationship],
    memories: list[RetrievedMemory],
) -> dict[str, tuple[float, str]]:
    desired: dict[str, tuple[float, str]] = {}
    needs = agent.needs
    food_stores = float(household.stored_resources.get("grain", 0)) + float(household.stored_resources.get("fish", 0))
    avg_trust = sum(rel.trust for rel in relationships) / len(relationships) if relationships else 0.4
    avg_affinity = sum(rel.affinity for rel in relationships) / len(relationships) if relationships else 0.4
    avg_reputation = sum(rel.reputation for rel in relationships) / len(relationships) if relationships else 0.5
    has_negative_memory = any(memory.polarity == "negative" for memory in memories)
    has_positive_memory = any(memory.polarity == "positive" for memory in memories)

    if needs.get("food", 100) < 32 or food_stores < 18:
        desired["secure_food"] = (0.62 + max(0, 32 - needs.get("food", 32)) / 100, "food need or household food pressure")
    if food_stores < 22:
        desired["maintain_household"] = (0.55 + max(0, 22 - food_stores) / 70, "low household food stores")
    if avg_trust < 0.42 or avg_affinity < 0.38 or has_negative_memory:
        desired["repair_relationship"] = (0.52 + (0.12 if has_negative_memory else 0.0), "strained relationship or conflict memory")
        desired["avoid_conflict"] = (0.5 + (0.16 if has_negative_memory else 0.0), "conflict risk is visible")
    if avg_trust > 0.55 or has_positive_memory:
        desired["seek_cooperation"] = (0.48 + (0.12 if has_positive_memory else 0.0), "trust or positive memory supports cooperation")
    if avg_reputation < 0.46:
        desired["improve_reputation"] = (0.52 + (0.46 - avg_reputation), "low relationship reputation")
    if needs.get("rest", 100) < 22:
        desired["rest_or_recover"] = (0.58 + max(0, 22 - needs.get("rest", 22)) / 100, "low rest need")
    return desired


def _next_goal_id(session: Session, run_id: str) -> str:
    count = session.query(Goal).filter(Goal.run_id == run_id).count()
    return f"{run_id}-goal-{count + 1:06d}"
