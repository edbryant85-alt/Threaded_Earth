from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.models import Event, Memory, Relationship


POSITIVE_EVENTS = {"cooperation", "resource_exchange"}
NEGATIVE_EVENTS = {"conflict"}
RESOURCE_STRESS_EVENTS = {"resource_stress", "resource_change", "resource_exchange", "conflict"}


@dataclass(frozen=True)
class RetrievedMemory:
    memory_id: str
    event_id: str
    event_type: str
    score: float
    salience: float
    recency: float
    polarity: str
    involved_agents: list[str]
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "score": self.score,
            "salience": self.salience,
            "recency": self.recency,
            "polarity": self.polarity,
            "involved_agents": self.involved_agents,
            "summary": self.summary,
        }


def retrieve_relevant_memories(
    session: Session,
    run_id: str,
    agent_id: str,
    tick: int,
    relationships: list[Relationship],
    limit: int = 3,
) -> list[RetrievedMemory]:
    relationship_targets = {relationship.target_agent for relationship in relationships}
    rows = (
        session.query(Memory, Event)
        .join(Event, Memory.event_id == Event.event_id)
        .filter(Memory.run_id == run_id, Memory.agent_id == agent_id, Memory.created_tick < tick)
        .order_by(Memory.created_tick.desc(), Memory.memory_id)
        .all()
    )
    scored = [
        _score_memory(memory, event, tick, relationship_targets)
        for memory, event in rows
    ]
    scored = [memory for memory in scored if memory.score > 0]
    return sorted(scored, key=lambda memory: (-memory.score, memory.memory_id))[:limit]


def memory_adjustments(retrieved: list[RetrievedMemory], avg_trust: float) -> dict[str, float]:
    adjustments = {
        "seek_food": 0.0,
        "cooperate": 0.0,
        "rest": 0.0,
        "share_food": 0.0,
        "conflict_over_food": 0.0,
        "avoid_conflict": 0.0,
        "repair_relationship": 0.0,
    }
    for memory in retrieved:
        weight = min(0.12, memory.score * 0.08)
        if memory.polarity == "positive":
            adjustments["cooperate"] += weight
            adjustments["repair_relationship"] += weight * 0.35
            adjustments["share_food"] += weight * 0.55
            adjustments["conflict_over_food"] -= weight * 0.45
            adjustments["avoid_conflict"] -= weight * 0.25
        elif memory.polarity == "negative":
            trust_buffer = 0.45 if avg_trust >= 0.68 else 1.0
            adjustments["cooperate"] -= weight * trust_buffer
            adjustments["repair_relationship"] += weight * 0.65
            adjustments["share_food"] -= weight * 0.45
            adjustments["conflict_over_food"] += weight * 0.75
            adjustments["avoid_conflict"] += weight * 0.85
        elif memory.polarity == "resource_stress":
            adjustments["seek_food"] += weight
            adjustments["share_food"] -= weight * 0.35
            adjustments["conflict_over_food"] += weight * 0.25
    return {key: round(value, 3) for key, value in adjustments.items() if abs(value) >= 0.001}


def summarize_memory_influence(retrieved: list[RetrievedMemory], adjustments: dict[str, float]) -> str:
    if not retrieved:
        return "No memories retrieved."
    types: dict[str, int] = {}
    for memory in retrieved:
        types[memory.event_type] = types.get(memory.event_type, 0) + 1
    adjustment_bits = ", ".join(f"{action} {delta:+.3f}" for action, delta in sorted(adjustments.items()))
    type_bits = ", ".join(f"{event_type}={count}" for event_type, count in sorted(types.items()))
    return f"Retrieved {len(retrieved)} memories ({type_bits}); score adjustments: {adjustment_bits or 'none'}."


def memory_stats(session: Session, run_id: str) -> dict[str, Any]:
    memories = session.query(Memory).filter(Memory.run_id == run_id).all()
    agent_count = len({memory.agent_id for memory in memories})
    high_salience = sum(1 for memory in memories if memory.salience >= 0.7)
    return {
        "total_memories": len(memories),
        "average_memories_per_agent": round(len(memories) / agent_count, 2) if agent_count else 0.0,
        "high_salience_memory_count": high_salience,
    }


def _score_memory(
    memory: Memory,
    event: Event,
    tick: int,
    relationship_targets: set[str],
) -> RetrievedMemory:
    recency = 1 / (1 + max(0, tick - memory.created_tick))
    involved = sorted({agent for agent in (event.actor, event.target) if agent})
    target_match = 1.0 if relationship_targets.intersection(involved) else 0.0
    event_weight = _event_weight(event)
    score = memory.salience * 0.55 + recency * 0.25 + target_match * 0.15 + event_weight * 0.2
    return RetrievedMemory(
        memory_id=memory.memory_id,
        event_id=memory.event_id,
        event_type=event.event_type,
        score=round(score, 4),
        salience=memory.salience,
        recency=round(recency, 4),
        polarity=_polarity(event),
        involved_agents=involved,
        summary=memory.summary,
    )


def _event_weight(event: Event) -> float:
    if event.event_type in {"conflict", "cooperation"}:
        return 1.0
    if event.event_type == "resource_exchange":
        return 0.85
    if event.event_type == "resource_change":
        return 0.65
    return 0.35


def _polarity(event: Event) -> str:
    if event.event_type in NEGATIVE_EVENTS:
        return "negative"
    if event.event_type in POSITIVE_EVENTS:
        delta = event.payload.get("resource_delta")
        if isinstance(delta, int | float) and delta < 0:
            return "resource_stress"
        return "positive"
    if event.event_type in RESOURCE_STRESS_EVENTS:
        return "resource_stress"
    return "neutral"
