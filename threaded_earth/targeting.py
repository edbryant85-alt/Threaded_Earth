from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.memory import RetrievedMemory
from threaded_earth.models import Agent, Decision, Goal, Household, Relationship, RoleSignal


SOCIAL_ACTIONS = {"cooperate", "share_food", "conflict_over_food", "avoid_conflict", "repair_relationship"}
HOSTILITY_THRESHOLD = 0.18


@dataclass(frozen=True)
class TargetSelection:
    selected_target_agent_id: str | None
    target_selection_candidates: list[dict[str, Any]]
    target_selection_scores: dict[str, float]
    target_selection_reasons: list[str]
    target_memory_factors: dict[str, Any]
    target_goal_factors: dict[str, Any]


@dataclass(frozen=True)
class TargetAwareActionEvaluation:
    target_aware_action_scores: dict[str, float]
    best_target_by_action: dict[str, Any]
    target_aware_score_reasons: list[str]
    selections_by_action: dict[str, TargetSelection]
    social_candidates_evaluated: int


def select_target_for_action(
    action: str,
    agent: Agent,
    relationships: list[Relationship],
    memories: list[RetrievedMemory],
    goals: list[Goal],
    households_by_agent: dict[str, Household] | None = None,
    role_signals_by_agent: dict[str, list[RoleSignal]] | None = None,
) -> TargetSelection:
    if action not in SOCIAL_ACTIONS or not relationships:
        return TargetSelection(None, [], {}, [], {}, {})

    scored = []
    for relationship in relationships:
        memory_factor = _memory_factor(relationship.target_agent, memories)
        goal_factor = _goal_factor(action, relationship, goals)
        kin_bonus = 0.16 if relationship.kinship_relation == "household kin" else 0.0
        resource_bonus = _resource_bonus(action, relationship.target_agent, households_by_agent)
        role_factor = _target_role_factor(action, relationship.target_agent, role_signals_by_agent)
        score = (
            _base_score(action, relationship)
            + kin_bonus
            + memory_factor["score"]
            + goal_factor["score"]
            + resource_bonus
            + role_factor["score"]
        )
        if action == "repair_relationship" and _hostility(relationship) < HOSTILITY_THRESHOLD:
            score -= 0.7
        scored.append(
            {
                "target_agent_id": relationship.target_agent,
                "score": round(score, 4),
                "trust": round(relationship.trust, 3),
                "affinity": round(relationship.affinity, 3),
                "reputation": round(relationship.reputation, 3),
                "kinship_relation": relationship.kinship_relation,
                "memory_factor": memory_factor,
                "goal_factor": goal_factor,
                "role_factor": role_factor,
                "resource_bonus": round(resource_bonus, 3),
            }
        )

    scored = sorted(scored, key=lambda item: (-item["score"], item["target_agent_id"]))
    selected = scored[0] if scored else None
    selected_id = selected["target_agent_id"] if selected else None
    scores = {item["target_agent_id"]: item["score"] for item in scored[:5]}
    reasons = _selection_reasons(action, selected)
    memory_factors = {
        item["target_agent_id"]: item["memory_factor"]
        for item in scored[:5]
        if item["memory_factor"]["score"] != 0
    }
    goal_factors = {
        item["target_agent_id"]: item["goal_factor"]
        for item in scored[:5]
        if item["goal_factor"]["score"] != 0
    }
    role_factors = {
        item["target_agent_id"]: item["role_factor"]
        for item in scored[:5]
        if item["role_factor"]["score"] != 0
    }
    return TargetSelection(
        selected_target_agent_id=selected_id,
        target_selection_candidates=scored[:5],
        target_selection_scores=scores,
        target_selection_reasons=reasons,
        target_memory_factors=memory_factors,
        target_goal_factors={**goal_factors, "_role_factors": role_factors} if role_factors else goal_factors,
    )


def evaluate_target_aware_actions(
    actions: list[str],
    agent: Agent,
    relationships: list[Relationship],
    memories: list[RetrievedMemory],
    goals: list[Goal],
    households_by_agent: dict[str, Household] | None = None,
    role_signals_by_agent: dict[str, list[RoleSignal]] | None = None,
) -> TargetAwareActionEvaluation:
    action_scores: dict[str, float] = {}
    best_by_action: dict[str, Any] = {}
    reasons: list[str] = []
    selections: dict[str, TargetSelection] = {}
    evaluated = 0
    for action in actions:
        if action not in SOCIAL_ACTIONS:
            continue
        selection = select_target_for_action(
            action, agent, relationships, memories, goals, households_by_agent, role_signals_by_agent
        )
        selections[action] = selection
        evaluated += len(selection.target_selection_candidates)
        best = selection.target_selection_candidates[0] if selection.target_selection_candidates else None
        if not best:
            action_scores[action] = 0.0
            continue
        contribution = _target_aware_contribution(action, best)
        action_scores[action] = contribution
        best_by_action[action] = {
            "target_agent_id": best["target_agent_id"],
            "target_score": best["score"],
            "target_aware_contribution": contribution,
            "trust": best["trust"],
            "affinity": best["affinity"],
            "reputation": best["reputation"],
            "kinship_relation": best["kinship_relation"],
            "memory_factor": best["memory_factor"],
            "goal_factor": best["goal_factor"],
            "role_factor": best["role_factor"],
            "resource_bonus": best["resource_bonus"],
        }
        reasons.extend(_target_aware_reasons(action, best, contribution))
    return TargetAwareActionEvaluation(
        target_aware_action_scores=action_scores,
        best_target_by_action=best_by_action,
        target_aware_score_reasons=reasons[:12],
        selections_by_action=selections,
        social_candidates_evaluated=evaluated,
    )


def target_aware_stats(session: Session, run_id: str) -> dict[str, Any]:
    decisions = session.query(Decision).filter(Decision.run_id == run_id).all()
    with_scores = [decision for decision in decisions if any(decision.target_aware_action_scores.values())]
    reason_counts: dict[str, int] = {}
    social_candidates = 0
    for decision in with_scores:
        social_candidates += sum(
            len(candidate.get("target_candidates", []))
            for candidate in decision.candidate_actions
            if candidate.get("target_candidates")
        )
        for reason in decision.target_aware_score_reasons:
            category = reason.split(":", 1)[0]
            reason_counts[category] = reason_counts.get(category, 0) + 1
    return {
        "decisions_with_target_aware_scores": len(with_scores),
        "social_candidates_evaluated": social_candidates,
        "common_target_aware_reasons": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:8]),
    }


def social_action_label(action: str, goals: list[Goal]) -> str:
    goal_types = {goal.goal_type for goal in goals}
    if action == "cooperate" and "repair_relationship" in goal_types:
        return "repair_relationship"
    if action == "cooperate":
        return "cooperate"
    if action == "share_food":
        return "trade_share"
    if action == "conflict_over_food":
        return "conflict"
    if action == "avoid_conflict":
        return "avoid"
    if action == "repair_relationship":
        return "repair_relationship"
    return action


def target_stats(session: Session, run_id: str) -> dict[str, Any]:
    decisions = session.query(Decision).filter(Decision.run_id == run_id).all()
    social = [decision for decision in decisions if decision.selected_action.get("action") in SOCIAL_ACTIONS]
    targeted = [decision for decision in social if decision.selected_target_agent_id]
    by_target: dict[str, int] = {}
    by_action: dict[str, int] = {}
    for decision in targeted:
        target = decision.selected_target_agent_id
        if target is not None:
            by_target[target] = by_target.get(target, 0) + 1
        action = decision.selected_action.get("action", "unknown")
        by_action[action] = by_action.get(action, 0) + 1
    return {
        "targeted_social_decisions": len(targeted),
        "untargeted_social_decisions": len(social) - len(targeted),
        "most_targeted_agents": dict(sorted(by_target.items(), key=lambda item: (-item[1], item[0]))[:8]),
        "social_actions_by_type": dict(sorted(by_action.items())),
    }


def _target_aware_contribution(action: str, best: dict[str, Any]) -> float:
    raw = max(0.0, float(best["score"]))
    if action == "conflict_over_food":
        return round(min(0.045, raw * 0.035), 3)
    if action == "avoid_conflict":
        return round(min(0.075, raw * 0.06), 3)
    if action == "repair_relationship":
        return round(min(0.09, raw * 0.07), 3)
    if action == "share_food":
        return round(min(0.11, raw * 0.08), 3)
    if action == "cooperate":
        return round(min(0.105, raw * 0.075), 3)
    return 0.0


def _target_aware_reasons(action: str, best: dict[str, Any], contribution: float) -> list[str]:
    reasons = [f"{action}:best_target={best['target_agent_id']} contribution={contribution:+.3f}"]
    if best["kinship_relation"] == "household kin":
        reasons.append(f"{action}:kinship")
    if best["trust"] >= 0.65:
        reasons.append(f"{action}:high_trust")
    if best["trust"] <= 0.35:
        reasons.append(f"{action}:low_trust")
    if best["memory_factor"]["positive"] > 0:
        reasons.append(f"{action}:positive_memory")
    if best["memory_factor"]["negative"] > 0:
        reasons.append(f"{action}:negative_memory")
    if best["goal_factor"]["goal_types"]:
        reasons.append(f"{action}:goal_{'+'.join(best['goal_factor']['goal_types'])}")
    if best.get("role_factor", {}).get("role_names"):
        reasons.append(f"{action}:target_role_{'+'.join(best['role_factor']['role_names'])}")
    if best["resource_bonus"] > 0:
        reasons.append(f"{action}:target_scarcity")
    return reasons


def _base_score(action: str, relationship: Relationship) -> float:
    trust = relationship.trust
    affinity = relationship.affinity
    reputation = relationship.reputation
    if action == "cooperate":
        return trust * 0.45 + affinity * 0.3 + reputation * 0.15
    if action == "share_food":
        return trust * 0.35 + affinity * 0.25 + reputation * 0.25
    if action == "repair_relationship":
        damaged = max(0.0, 0.58 - ((trust + affinity) / 2))
        return damaged * 0.9 + reputation * 0.1
    if action == "avoid_conflict":
        return (1 - trust) * 0.45 + (1 - affinity) * 0.25 + max(0.0, 0.45 - reputation) * 0.3
    if action == "conflict_over_food":
        return (1 - trust) * 0.35 + (1 - reputation) * 0.3 + max(0.0, 0.35 - affinity) * 0.2
    return 0.0


def _memory_factor(target_agent: str, memories: list[RetrievedMemory]) -> dict[str, Any]:
    positive = 0.0
    negative = 0.0
    resource = 0.0
    memory_ids = []
    for memory in memories:
        if target_agent not in memory.involved_agents:
            continue
        memory_ids.append(memory.memory_id)
        if memory.polarity == "positive":
            positive += min(0.18, memory.score * 0.1)
        elif memory.polarity == "negative":
            negative += min(0.2, memory.score * 0.12)
        elif memory.polarity == "resource_stress":
            resource += min(0.1, memory.score * 0.06)
    return {
        "score": round(positive - negative + resource, 4),
        "positive": round(positive, 4),
        "negative": round(negative, 4),
        "resource": round(resource, 4),
        "memory_ids": memory_ids[:3],
    }


def _goal_factor(action: str, relationship: Relationship, goals: list[Goal]) -> dict[str, Any]:
    score = 0.0
    applied = []
    for goal in goals:
        weight = min(0.12, goal.priority * 0.08)
        if action == "cooperate" and goal.goal_type in {"seek_cooperation", "improve_reputation"}:
            score += weight
            applied.append(goal.goal_type)
        elif action == "share_food" and goal.goal_type in {"improve_reputation", "maintain_household", "seek_cooperation"}:
            score += weight * (1.15 if relationship.kinship_relation == "household kin" else 0.85)
            applied.append(goal.goal_type)
        elif action == "repair_relationship" and goal.goal_type == "repair_relationship":
            score += weight + max(0.0, 0.55 - relationship.trust) * 0.16
            applied.append(goal.goal_type)
        elif action == "avoid_conflict" and goal.goal_type == "avoid_conflict":
            score += weight + max(0.0, 0.45 - relationship.trust) * 0.16
            applied.append(goal.goal_type)
        elif action == "conflict_over_food" and goal.goal_type == "secure_food":
            score += weight * max(0.0, 0.45 - relationship.trust)
            applied.append(goal.goal_type)
    return {"score": round(score, 4), "goal_types": applied}


def _resource_bonus(
    action: str,
    target_agent: str,
    households_by_agent: dict[str, Household] | None,
) -> float:
    if action != "share_food" or not households_by_agent or target_agent not in households_by_agent:
        return 0.0
    household = households_by_agent[target_agent]
    food = float(household.stored_resources.get("grain", 0)) + float(household.stored_resources.get("fish", 0))
    return min(0.14, max(0.0, 18 - food) / 100)


def _target_role_factor(
    action: str,
    target_agent: str,
    role_signals_by_agent: dict[str, list[RoleSignal]] | None,
) -> dict[str, Any]:
    if not role_signals_by_agent:
        return {"score": 0.0, "role_names": []}
    score = 0.0
    applied = []
    for role in role_signals_by_agent.get(target_agent, []):
        if role.role_name == "trusted_neighbor" and action in {"cooperate", "share_food"}:
            score += min(0.06, role.score * 0.06)
            applied.append(role.role_name)
        elif role.role_name == "helper" and action == "cooperate":
            score += min(0.025, role.score * 0.025)
            applied.append(role.role_name)
        elif role.role_name == "conflict_prone" and action in {"avoid_conflict", "conflict_over_food"}:
            score += min(0.035, role.score * 0.04)
            applied.append(role.role_name)
    return {"score": round(score, 4), "role_names": applied[:3]}


def _hostility(relationship: Relationship) -> float:
    return (relationship.trust + relationship.affinity) / 2


def _selection_reasons(action: str, selected: dict[str, Any] | None) -> list[str]:
    if not selected:
        return [f"no target selected for {action}"]
    return [
        f"action={action}",
        f"selected={selected['target_agent_id']}",
        f"score={selected['score']}",
        f"trust={selected['trust']}",
        f"affinity={selected['affinity']}",
        f"reputation={selected['reputation']}",
        f"kinship={selected['kinship_relation']}",
    ]
