from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from threaded_earth.goals import goal_adjustments, summarize_goal_influence
from threaded_earth.memory import RetrievedMemory, memory_adjustments, summarize_memory_influence
from threaded_earth.models import Agent, Goal, Household, Relationship
from threaded_earth.targeting import select_target_for_action


@dataclass(frozen=True)
class DecisionTrace:
    candidate_actions: list[dict[str, Any]]
    selected_action: dict[str, Any]
    reasons: list[str]
    confidence: float
    uncertainty_notes: str
    relationship_modifiers: dict[str, Any]
    active_needs: dict[str, Any]
    memories_consulted: list[str]
    retrieved_memory_ids: list[str]
    memory_influence_summary: str
    memory_score_adjustments: dict[str, float]
    active_goal_ids: list[str]
    goal_influence_summary: str
    goal_score_adjustments: dict[str, float]
    selected_target_agent_id: str | None
    target_selection_candidates: list[dict[str, Any]]
    target_selection_scores: dict[str, float]
    target_selection_reasons: list[str]
    target_memory_factors: dict[str, Any]
    target_goal_factors: dict[str, Any]


def choose_action(
    agent: Agent,
    household: Household,
    relationships: list[Relationship],
    rng: random.Random,
    tick: int,
    retrieved_memories: list[RetrievedMemory] | None = None,
    active_goals: list[Goal] | None = None,
    households_by_agent: dict[str, Household] | None = None,
) -> DecisionTrace:
    retrieved_memories = retrieved_memories or []
    active_goals = active_goals or []
    needs = dict(agent.needs)
    food_stores = float(household.stored_resources.get("grain", 0)) + float(household.stored_resources.get("fish", 0))
    avg_trust = sum(rel.trust for rel in relationships) / len(relationships) if relationships else 0.35
    avg_affinity = sum(rel.affinity for rel in relationships) / len(relationships) if relationships else 0.35
    pressure = max(0.0, (35 - needs["food"]) / 35) + max(0.0, (18 - food_stores) / 18)

    candidates = [
        _candidate("seek_food", 0.28 + pressure, "raises household food stores"),
        _candidate("cooperate", 0.22 + avg_trust * 0.45 + needs["belonging"] / 180, "strengthens a nearby tie"),
        _candidate("rest", 0.2 + max(0, 30 - needs["rest"]) / 80, "restores personal energy"),
        _candidate("share_food", 0.15 + avg_affinity * 0.35 if food_stores > 14 else 0.05, "transfers food to a strained household"),
        _candidate("repair_relationship", 0.08 + max(0, 0.5 - avg_trust) * 0.18, "repairs a strained social tie"),
        _candidate("avoid_conflict", 0.06 + max(0, 0.48 - avg_trust) * 0.2, "keeps distance from a risky tie"),
        _candidate("conflict_over_food", 0.05 + pressure * 0.28 + max(0, 0.45 - avg_trust) * 0.3, "contests scarce food"),
    ]
    if agent.archetype in {"cultivator", "gatherer", "fisher", "hunter"}:
        candidates[0]["score"] += 0.18
    if agent.archetype in {"healer", "elder", "storyteller"}:
        candidates[1]["score"] += 0.15
    if agent.archetype in {"trader", "craft worker", "builder"}:
        candidates[3]["score"] += 0.08

    adjustments = memory_adjustments(retrieved_memories, avg_trust)
    goal_scores = goal_adjustments(active_goals)
    for candidate in candidates:
        candidate["base_score"] = round(candidate["score"], 3)
        candidate["memory_adjustment"] = adjustments.get(candidate["action"], 0.0)
        candidate["goal_adjustment"] = goal_scores.get(candidate["action"], 0.0)
        candidate["score"] = round(
            max(0.01, candidate["score"] + candidate["memory_adjustment"] + candidate["goal_adjustment"]),
            3,
        )

    total = sum(max(0.01, candidate["score"]) for candidate in candidates)
    pick = rng.uniform(0, total)
    running = 0.0
    selected = candidates[-1]
    for candidate in candidates:
        running += max(0.01, candidate["score"])
        if running >= pick:
            selected = candidate
            break

    target_selection = select_target_for_action(
        selected["action"],
        agent,
        relationships,
        retrieved_memories,
        active_goals,
        households_by_agent,
    )

    sorted_scores = sorted((candidate["score"] for candidate in candidates), reverse=True)
    confidence = min(0.95, max(0.35, sorted_scores[0] / max(0.01, total) + 0.32))
    reasons = [
        f"food_need={needs['food']}",
        f"rest_need={needs['rest']}",
        f"belonging_need={needs['belonging']}",
        f"household_food={food_stores:.1f}",
        f"avg_trust={avg_trust:.2f}",
        f"tick={tick}",
    ]
    if retrieved_memories:
        reasons.append(f"retrieved_memory_ids={[memory.memory_id for memory in retrieved_memories]}")
        reasons.append(f"memory_score_adjustments={adjustments}")
    if active_goals:
        reasons.append(f"active_goal_ids={[goal.goal_id for goal in active_goals]}")
        reasons.append(f"goal_score_adjustments={goal_scores}")
    if target_selection.selected_target_agent_id:
        reasons.append(f"selected_target_agent_id={target_selection.selected_target_agent_id}")
        reasons.append(f"target_selection_scores={target_selection.target_selection_scores}")
    return DecisionTrace(
        candidate_actions=candidates,
        selected_action={
            "action": selected["action"],
            "score": round(selected["score"], 3),
            "selected_target_agent_id": target_selection.selected_target_agent_id,
        },
        reasons=reasons,
        confidence=round(confidence, 3),
        uncertainty_notes="Local symbolic rule choice; no hidden model or LLM inference used.",
        relationship_modifiers={"avg_trust": round(avg_trust, 3), "avg_affinity": round(avg_affinity, 3)},
        active_needs=needs,
        memories_consulted=[memory.memory_id for memory in retrieved_memories],
        retrieved_memory_ids=[memory.memory_id for memory in retrieved_memories],
        memory_influence_summary=summarize_memory_influence(retrieved_memories, adjustments),
        memory_score_adjustments=adjustments,
        active_goal_ids=[goal.goal_id for goal in active_goals],
        goal_influence_summary=summarize_goal_influence(active_goals, goal_scores),
        goal_score_adjustments=goal_scores,
        selected_target_agent_id=target_selection.selected_target_agent_id,
        target_selection_candidates=target_selection.target_selection_candidates,
        target_selection_scores=target_selection.target_selection_scores,
        target_selection_reasons=target_selection.target_selection_reasons,
        target_memory_factors=target_selection.target_memory_factors,
        target_goal_factors=target_selection.target_goal_factors,
    )


def _candidate(action: str, score: float, rationale: str) -> dict[str, Any]:
    return {"action": action, "score": round(score, 3), "rationale": rationale}
