from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from threaded_earth.models import Agent, Household, Relationship


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


def choose_action(
    agent: Agent,
    household: Household,
    relationships: list[Relationship],
    rng: random.Random,
    tick: int,
) -> DecisionTrace:
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
        _candidate("conflict_over_food", 0.05 + pressure * 0.28 + max(0, 0.45 - avg_trust) * 0.3, "contests scarce food"),
    ]
    if agent.archetype in {"cultivator", "gatherer", "fisher", "hunter"}:
        candidates[0]["score"] += 0.18
    if agent.archetype in {"healer", "elder", "storyteller"}:
        candidates[1]["score"] += 0.15
    if agent.archetype in {"trader", "craft worker", "builder"}:
        candidates[3]["score"] += 0.08

    total = sum(max(0.01, candidate["score"]) for candidate in candidates)
    pick = rng.uniform(0, total)
    running = 0.0
    selected = candidates[-1]
    for candidate in candidates:
        running += max(0.01, candidate["score"])
        if running >= pick:
            selected = candidate
            break

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
    return DecisionTrace(
        candidate_actions=candidates,
        selected_action={"action": selected["action"], "score": round(selected["score"], 3)},
        reasons=reasons,
        confidence=round(confidence, 3),
        uncertainty_notes="Local symbolic rule choice; no hidden model or LLM inference used.",
        relationship_modifiers={"avg_trust": round(avg_trust, 3), "avg_affinity": round(avg_affinity, 3)},
        active_needs=needs,
        memories_consulted=[],
    )


def _candidate(action: str, score: float, rationale: str) -> dict[str, Any]:
    return {"action": action, "score": round(score, 3), "rationale": rationale}
