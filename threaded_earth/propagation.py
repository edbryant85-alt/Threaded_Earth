from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.config import PropagationConfig
from threaded_earth.events import record_event
from threaded_earth.models import Agent, Event, Household, Memory, Relationship


ELIGIBLE_PROPAGATION_EVENTS = {"cooperation", "resource_exchange", "conflict", "repair", "household_shortage"}


@dataclass(frozen=True)
class PropagationCandidate:
    observer_agent_id: str
    subject_agent_id: str | None
    reason: str
    rank: int


def propagate_social_event(
    session: Session,
    run_id: str,
    tick: int,
    source_event: Event,
    agents_by_id: dict[str, Agent],
    households_by_agent: dict[str, Household],
    config: PropagationConfig,
) -> list[Event]:
    if not config.propagation_enabled or source_event.event_type not in ELIGIBLE_PROPAGATION_EVENTS:
        return []

    candidates = _select_observers(session, run_id, source_event, agents_by_id, households_by_agent, config)
    emitted: list[Event] = []
    for candidate in candidates[: config.propagation_max_observers]:
        relationship_delta = _apply_observer_effect(
            session,
            run_id,
            source_event.event_type,
            candidate.observer_agent_id,
            candidate.subject_agent_id,
            config.propagation_strength,
        )
        memory_id = None
        if config.propagation_create_memories:
            memory_id = _create_propagated_memory(
                session,
                run_id,
                tick,
                candidate.observer_agent_id,
                source_event,
                config.propagation_memory_salience_multiplier,
                candidate.reason,
            )
        payload = {
            "source_event_id": source_event.event_id,
            "observer_agent_id": candidate.observer_agent_id,
            "subject_agent_id": candidate.subject_agent_id,
            "propagation_reason": candidate.reason,
            "propagation_strength": config.propagation_strength,
            "memory_id": memory_id,
            "relationship_delta": relationship_delta,
            "depth": 1,
        }
        summary = (
            f"{candidate.observer_agent_id} indirectly registered {source_event.event_type} "
            f"involving {candidate.subject_agent_id or source_event.target}."
        )
        emitted.append(
            record_event(
                session,
                run_id,
                tick,
                "social_propagation",
                candidate.subject_agent_id,
                candidate.observer_agent_id,
                payload,
                summary,
            )
        )
    return emitted


def propagation_stats(session: Session, run_id: str) -> dict[str, Any]:
    events = _propagation_events(session, run_id)
    memories = _propagated_memories(session, run_id)
    reason_counts: dict[str, int] = {}
    subject_counts: dict[str, int] = {}
    for event in events:
        reason = event.payload.get("propagation_reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        subject = event.payload.get("subject_agent_id")
        if subject:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
    return {
        "propagated_event_count": len(events),
        "propagated_memory_count": len(memories),
        "common_propagation_reasons": _top_counts(reason_counts),
        "most_socially_visible_agents": _top_counts(subject_counts),
    }


def propagation_stats_for_tick(session: Session, run_id: str, tick: int) -> dict[str, Any]:
    events = (
        session.query(Event)
        .filter(Event.run_id == run_id, Event.tick == tick, Event.event_type == "social_propagation")
        .order_by(Event.event_id)
        .all()
    )
    observers = sorted({event.payload.get("observer_agent_id") for event in events if event.payload.get("observer_agent_id")})
    subjects: dict[str, int] = {}
    memories = 0
    for event in events:
        if event.payload.get("memory_id"):
            memories += 1
        subject = event.payload.get("subject_agent_id")
        if subject:
            subjects[subject] = subjects.get(subject, 0) + 1
    return {
        "propagated_events_this_tick": len(events),
        "propagated_memories_this_tick": memories,
        "observers_reached_this_tick": len(observers),
        "top_subjects_by_propagation": _top_counts(subjects),
    }


def recent_propagation_events(session: Session, run_id: str, limit: int = 8) -> list[Event]:
    return (
        session.query(Event)
        .filter(Event.run_id == run_id, Event.event_type == "social_propagation")
        .order_by(Event.tick.desc(), Event.event_id.desc())
        .limit(limit)
        .all()
    )


def _select_observers(
    session: Session,
    run_id: str,
    source_event: Event,
    agents_by_id: dict[str, Agent],
    households_by_agent: dict[str, Household],
    config: PropagationConfig,
) -> list[PropagationCandidate]:
    candidates: dict[str, PropagationCandidate] = {}
    actor_id = source_event.actor if source_event.actor in agents_by_id else None
    target_id = source_event.target if source_event.target in agents_by_id else None
    excluded = {agent_id for agent_id in [actor_id, target_id] if agent_id}

    if actor_id:
        _add_household_candidates(candidates, actor_id, households_by_agent, excluded, "actor_household", 10)
    if target_id:
        _add_household_candidates(candidates, target_id, households_by_agent, excluded, "target_household", 20)
    for subject_id in [actor_id, target_id]:
        if subject_id:
            _add_close_tie_candidates(session, run_id, candidates, subject_id, excluded, config, rank=30)

    if source_event.event_type == "household_shortage":
        household_id = source_event.target
        household = session.get(Household, household_id) if household_id else None
        household_members = set(household.members if household else [])
        for member_id in sorted(household_members):
            _add_close_tie_candidates(
                session,
                run_id,
                candidates,
                member_id,
                household_members,
                config,
                rank=40,
                subject_override=member_id,
                reason="close_tie_to_shortage",
            )

    return sorted(candidates.values(), key=lambda item: (item.rank, item.observer_agent_id, item.subject_agent_id or ""))


def _add_household_candidates(
    candidates: dict[str, PropagationCandidate],
    subject_id: str,
    households_by_agent: dict[str, Household],
    excluded: set[str],
    reason: str,
    rank: int,
) -> None:
    household = households_by_agent.get(subject_id)
    if household is None:
        return
    for member_id in sorted(household.members):
        if member_id in excluded:
            continue
        _set_candidate(candidates, member_id, subject_id, reason, rank)


def _add_close_tie_candidates(
    session: Session,
    run_id: str,
    candidates: dict[str, PropagationCandidate],
    subject_id: str,
    excluded: set[str],
    config: PropagationConfig,
    rank: int,
    subject_override: str | None = None,
    reason: str = "close_tie",
) -> None:
    relationships = (
        session.query(Relationship)
        .filter(Relationship.run_id == run_id, Relationship.source_agent == subject_id)
        .order_by(Relationship.target_agent)
        .all()
    )
    for relationship in relationships:
        observer_id = relationship.target_agent
        if observer_id in excluded:
            continue
        is_close_kin = relationship.kinship_relation in {"household", "kin"}
        is_high_trust = relationship.trust >= config.propagation_min_relationship_threshold
        is_high_affinity = relationship.affinity >= config.propagation_min_relationship_threshold
        if is_close_kin or is_high_trust or is_high_affinity:
            candidate_reason = reason if reason != "close_tie" else _close_tie_reason(relationship, config)
            _set_candidate(candidates, observer_id, subject_override or subject_id, candidate_reason, rank)


def _set_candidate(
    candidates: dict[str, PropagationCandidate],
    observer_id: str,
    subject_id: str | None,
    reason: str,
    rank: int,
) -> None:
    current = candidates.get(observer_id)
    if current is None or rank < current.rank:
        candidates[observer_id] = PropagationCandidate(observer_id, subject_id, reason, rank)


def _close_tie_reason(relationship: Relationship, config: PropagationConfig) -> str:
    if relationship.kinship_relation in {"household", "kin"}:
        return "close_kin_tie"
    if relationship.trust >= config.propagation_min_relationship_threshold:
        return "high_trust_tie"
    return "high_affinity_tie"


def _apply_observer_effect(
    session: Session,
    run_id: str,
    event_type: str,
    observer_agent_id: str,
    subject_agent_id: str | None,
    strength: float,
) -> dict[str, float]:
    if subject_agent_id is None or observer_agent_id == subject_agent_id:
        return {}
    relationship = (
        session.query(Relationship)
        .filter(
            Relationship.run_id == run_id,
            Relationship.source_agent == observer_agent_id,
            Relationship.target_agent == subject_agent_id,
        )
        .first()
    )
    if relationship is None:
        return {}
    direct = _direct_delta_for_event(event_type)
    applied: dict[str, float] = {}
    for key, value in direct.items():
        delta = round(value * strength, 4)
        if key == "trust":
            relationship.trust = _clamp(relationship.trust + delta)
        elif key == "affinity":
            relationship.affinity = _clamp(relationship.affinity + delta)
        elif key == "reputation":
            relationship.reputation = _clamp(relationship.reputation + delta)
        applied[key] = delta
    return applied


def _direct_delta_for_event(event_type: str) -> dict[str, float]:
    if event_type == "cooperation":
        return {"trust": 0.04, "affinity": 0.03, "reputation": 0.02}
    if event_type == "resource_exchange":
        return {"trust": 0.03, "reputation": 0.05}
    if event_type == "repair":
        return {"trust": 0.05, "affinity": 0.04, "reputation": 0.02}
    if event_type == "conflict":
        return {"trust": -0.06, "affinity": -0.05, "reputation": -0.03}
    return {}


def _create_propagated_memory(
    session: Session,
    run_id: str,
    tick: int,
    observer_agent_id: str,
    source_event: Event,
    salience_multiplier: float,
    reason: str,
) -> str:
    count = session.query(Memory).filter(Memory.run_id == run_id).count()
    salience = round(_base_salience(source_event.event_type) * salience_multiplier, 3)
    memory_id = f"{run_id}-mem-{count + 1:06d}"
    session.add(
        Memory(
            memory_id=memory_id,
            run_id=run_id,
            agent_id=observer_agent_id,
            event_id=source_event.event_id,
            salience=salience,
            summary=f"Indirect social memory ({reason}): {source_event.summary}",
            created_tick=tick,
        )
    )
    session.flush()
    return memory_id


def _base_salience(event_type: str) -> float:
    return {
        "cooperation": 0.58,
        "resource_exchange": 0.64,
        "repair": 0.66,
        "conflict": 0.78,
        "household_shortage": 0.7,
    }.get(event_type, 0.55)


def _propagation_events(session: Session, run_id: str) -> list[Event]:
    return (
        session.query(Event)
        .filter(Event.run_id == run_id, Event.event_type == "social_propagation")
        .order_by(Event.tick, Event.event_id)
        .all()
    )


def _propagated_memories(session: Session, run_id: str) -> list[Memory]:
    return (
        session.query(Memory)
        .filter(Memory.run_id == run_id, Memory.summary.like("Indirect social memory%"))
        .order_by(Memory.created_tick, Memory.memory_id)
        .all()
    )


def _top_counts(counts: dict[str, int], limit: int = 5) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)
