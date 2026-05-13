from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.config import NormConfig
from threaded_earth.models import Agent, Event, NormCandidate, Relationship


STABLE_EVIDENCE_THRESHOLD = 4


def update_norm_candidates_for_event(
    session: Session,
    run_id: str,
    event: Event,
    actor: Agent | None,
    target: Agent | None,
    relationship: Relationship | None,
    tick: int,
    config: NormConfig | None = None,
) -> None:
    for norm_name, support, opposition, summary in _evidence_from_event(event, actor, target, relationship):
        update_norm_candidate(
            session,
            run_id,
            norm_name,
            support,
            opposition,
            tick,
            summary,
            actor_id=actor.neutral_id if actor else event.actor,
            household_id=actor.household_id if actor else None,
            event_id=event.event_id,
            config=config,
        )


def update_norm_candidate(
    session: Session,
    run_id: str,
    norm_name: str,
    support: float,
    opposition: float,
    tick: int,
    evidence_summary: str,
    actor_id: str | None = None,
    household_id: str | None = None,
    event_id: str | None = None,
    config: NormConfig | None = None,
) -> NormCandidate:
    config = config or NormConfig()
    norm = (
        session.query(NormCandidate)
        .filter(NormCandidate.run_id == run_id, NormCandidate.norm_name == norm_name)
        .first()
    )
    if norm is None:
        count = session.query(NormCandidate).filter(NormCandidate.run_id == run_id).count()
        norm = NormCandidate(
            norm_candidate_id=f"{run_id}-norm-{count + 1:06d}",
            run_id=run_id,
            norm_name=norm_name,
            evidence_count=0,
            support_score=0.0,
            opposition_score=0.0,
            contributing_agent_ids=[],
            contributing_household_ids=[],
            contributing_event_ids=[],
            recent_tick_count=0,
            evidence_density=0.0,
            breadth_score=0.0,
            first_observed_tick=tick,
            last_observed_tick=tick,
            status="emerging",
            evidence_summary=evidence_summary,
        )
        session.add(norm)
    agent_ids = list(norm.contributing_agent_ids or [])
    household_ids = list(norm.contributing_household_ids or [])
    event_ids = list(norm.contributing_event_ids or [])
    actor_factor = 1.0 if actor_id is None or actor_id not in agent_ids else config.norm_repeated_actor_diminishing_factor
    household_factor = (
        1.0
        if household_id is None or household_id not in household_ids
        else config.norm_repeated_household_diminishing_factor
    )
    contribution_factor = actor_factor * household_factor
    if actor_id and actor_id not in agent_ids:
        agent_ids.append(actor_id)
    if household_id and household_id not in household_ids:
        household_ids.append(household_id)
    if event_id and event_id not in event_ids:
        event_ids.append(event_id)
    norm.evidence_count += 1
    norm.support_score = round(max(0.0, norm.support_score + support * contribution_factor), 3)
    norm.opposition_score = round(max(0.0, norm.opposition_score + opposition * contribution_factor), 3)
    norm.contributing_agent_ids = agent_ids
    norm.contributing_household_ids = household_ids
    norm.contributing_event_ids = event_ids
    norm.last_observed_tick = tick
    norm.recent_tick_count = max(1, norm.last_observed_tick - norm.first_observed_tick + 1)
    norm.evidence_density = round(norm.evidence_count / norm.recent_tick_count, 3)
    norm.breadth_score = _breadth_score(norm)
    norm.evidence_summary = evidence_summary
    norm.status = _status(norm, config)
    session.flush()
    return norm


def norm_stats(session: Session, run_id: str) -> dict[str, Any]:
    norms = (
        session.query(NormCandidate)
        .filter(NormCandidate.run_id == run_id)
        .order_by(NormCandidate.support_score.desc(), NormCandidate.evidence_count.desc(), NormCandidate.norm_name)
        .all()
    )
    by_status = {"emerging": 0, "stable": 0, "declining": 0}
    for norm in norms:
        by_status[norm.status] = by_status.get(norm.status, 0) + 1
    return {
        "norm_candidates_total": len(norms),
        "emerging_norms": by_status.get("emerging", 0),
        "stable_norms": by_status.get("stable", 0),
        "declining_norms": by_status.get("declining", 0),
        "top_norm_candidates": [_norm_payload(norm) for norm in norms[:8]],
    }


def recent_norm_candidates(session: Session, run_id: str, limit: int = 12) -> list[NormCandidate]:
    return (
        session.query(NormCandidate)
        .filter(NormCandidate.run_id == run_id)
        .order_by(NormCandidate.support_score.desc(), NormCandidate.last_observed_tick.desc(), NormCandidate.norm_name)
        .limit(limit)
        .all()
    )


def _evidence_from_event(
    event: Event,
    actor: Agent | None,
    target: Agent | None,
    relationship: Relationship | None,
) -> list[tuple[str, float, float, str]]:
    evidence: list[tuple[str, float, float, str]] = []
    if event.event_type == "resource_exchange":
        transfer = event.payload.get("resource_transfer") or {}
        success = float(transfer.get("transferred_quantity") or 0) > 0
        is_kin = relationship is not None and relationship.kinship_relation == "household kin"
        if success and is_kin:
            evidence.append(("sharing_food_with_kin", 0.35, 0.0, f"shared with kin: {event.event_id}"))
        elif not success and is_kin:
            evidence.append(("sharing_food_with_kin", 0.0, 0.2, f"failed kin sharing: {event.event_id}"))
        if success:
            evidence.append(("supporting_households_in_scarcity", 0.15, 0.0, f"resource support: {event.event_id}"))
        else:
            evidence.append(("supporting_households_in_scarcity", 0.0, 0.15, f"insufficient sharing: {event.event_id}"))
    elif event.event_type == "cooperation":
        if relationship is not None and relationship.trust >= 0.6:
            evidence.append(("helping_trusted_neighbors", 0.25, 0.0, f"helped trusted target: {event.event_id}"))
        if actor and actor.archetype in {"cultivator", "gatherer", "fisher", "hunter", "builder", "craft worker"}:
            evidence.append(("valuing_providers", 0.12, 0.0, f"provider helped: {event.event_id}"))
    elif event.event_type == "repair":
        evidence.append(("repairing_after_conflict", 0.3, 0.0, f"repair attempt: {event.event_id}"))
        evidence.append(("valuing_mediators", 0.16, 0.0, f"mediating action: {event.event_id}"))
    elif event.event_type == "avoidance":
        if relationship is not None and relationship.trust <= 0.45:
            evidence.append(("avoiding_low_trust_agents", 0.22, 0.0, f"avoided low-trust tie: {event.event_id}"))
        elif relationship is not None and relationship.trust >= 0.65:
            evidence.append(("helping_trusted_neighbors", 0.0, 0.12, f"avoided trusted tie: {event.event_id}"))
    elif event.event_type == "conflict":
        evidence.append(("disfavoring_repeated_conflict", 0.25, 0.0, f"conflict observed: {event.event_id}"))
        evidence.append(("repairing_after_conflict", 0.0, 0.14, f"conflict without repair: {event.event_id}"))
    return evidence


def _status(norm: NormCandidate, config: NormConfig) -> str:
    net_support = norm.support_score - norm.opposition_score
    if norm.opposition_score > norm.support_score + config.norm_decline_threshold:
        return "declining"
    if (
        norm.evidence_count >= STABLE_EVIDENCE_THRESHOLD
        and net_support >= config.norm_stability_support_threshold
        and len(norm.contributing_agent_ids or []) >= config.norm_min_agents_for_stable
        and len(norm.contributing_household_ids or []) >= config.norm_min_households_for_stable
    ):
        return "stable"
    return "emerging"


def _breadth_score(norm: NormCandidate) -> float:
    agent_count = len(norm.contributing_agent_ids or [])
    household_count = len(norm.contributing_household_ids or [])
    event_count = max(1, len(norm.contributing_event_ids or []))
    return round(((agent_count * 0.6) + (household_count * 0.9)) / event_count, 3)


def _norm_payload(norm: NormCandidate) -> dict[str, Any]:
    return {
        "norm_candidate_id": norm.norm_candidate_id,
        "norm_name": norm.norm_name,
        "evidence_count": norm.evidence_count,
        "contributing_agent_count": len(norm.contributing_agent_ids or []),
        "contributing_household_count": len(norm.contributing_household_ids or []),
        "contributing_event_count": len(norm.contributing_event_ids or []),
        "recent_tick_count": norm.recent_tick_count,
        "evidence_density": round(norm.evidence_density, 3),
        "breadth_score": round(norm.breadth_score, 3),
        "support_score": round(norm.support_score, 3),
        "opposition_score": round(norm.opposition_score, 3),
        "status": norm.status,
        "first_observed_tick": norm.first_observed_tick,
        "last_observed_tick": norm.last_observed_tick,
        "evidence_summary": norm.evidence_summary,
    }
