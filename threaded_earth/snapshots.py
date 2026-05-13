from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.goals import goal_stats
from threaded_earth.memory import memory_stats
from threaded_earth.metrics import compute_metrics
from threaded_earth.models import Agent, Event, Household, Relationship, Resource
from threaded_earth.paths import snapshot_path, snapshots_dir
from threaded_earth.propagation import propagation_stats_for_tick
from threaded_earth.resources import household_resource_summary, transfers_for_tick, upkeep_stats_for_tick
from threaded_earth.roles import role_stats
from threaded_earth.targeting import target_aware_stats, target_stats


SNAPSHOT_RE = re.compile(r"^tick_(\d+)\.json$")
DELTA_METRICS = [
    "relationship_density",
    "conflict_frequency",
    "cooperation_frequency",
    "resource_stress",
    "reputation_variance",
]


@dataclass(frozen=True)
class SnapshotInventory:
    status: str
    count: int
    expected_ticks: int | None
    ticks: list[int]
    latest_tick: int | None


def write_snapshot(session: Session, run_id: str, tick: int) -> Path:
    snapshot = build_snapshot(session, run_id, tick)
    path = snapshot_path(run_id, tick)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_snapshot(session: Session, run_id: str, tick: int) -> dict[str, Any]:
    events = session.query(Event).filter(Event.run_id == run_id, Event.tick == tick).order_by(Event.event_id).all()
    agents = session.query(Agent).filter(Agent.run_id == run_id).order_by(Agent.neutral_id).all()
    households = session.query(Household).filter(Household.run_id == run_id).order_by(Household.household_id).all()
    relationships = session.query(Relationship).filter(Relationship.run_id == run_id).all()
    resources = session.query(Resource).filter(Resource.run_id == run_id).order_by(Resource.owner_id, Resource.resource_type).all()
    return {
        "run_id": run_id,
        "tick": tick,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "agents_summary": _agents_summary(agents),
        "households_summary": _households_summary(households),
        "relationships_summary": _relationships_summary(relationships),
        "resources_summary": _resources_summary(resources),
        "household_resource_summary": household_resource_summary(session, run_id),
        "resource_transfers_this_tick": transfers_for_tick(session, run_id, tick),
        "upkeep_summary": {
            **upkeep_stats_for_tick(session, run_id, tick),
            "post_upkeep_total_food": household_resource_summary(session, run_id)["total_food"],
            "post_upkeep_total_materials": household_resource_summary(session, run_id)["total_materials"],
        },
        "memory_summary": memory_stats(session, run_id),
        "goal_summary": goal_stats(session, run_id),
        "target_summary": target_stats(session, run_id),
        "target_aware_summary": target_aware_stats(session, run_id),
        "propagation_summary": propagation_stats_for_tick(session, run_id, tick),
        "role_summary": role_stats(session, run_id),
        "metrics": compute_metrics(session, run_id),
        "event_ids": [event.event_id for event in events],
    }


def list_snapshot_ticks(run_id: str) -> list[int]:
    directory = snapshots_dir(run_id)
    if not directory.exists():
        return []
    ticks = []
    for path in directory.iterdir():
        match = SNAPSHOT_RE.match(path.name)
        if match:
            ticks.append(int(match.group(1)))
    return sorted(ticks)


def load_snapshot(run_id: str, tick: int) -> dict[str, Any]:
    return json.loads(snapshot_path(run_id, tick).read_text(encoding="utf-8"))


def load_snapshots(run_id: str) -> list[dict[str, Any]]:
    return [load_snapshot(run_id, tick) for tick in list_snapshot_ticks(run_id)]


def expected_tick_count(session: Session, run_id: str) -> int | None:
    max_tick = session.query(Event.tick).filter(Event.run_id == run_id).order_by(Event.tick.desc()).limit(1).scalar()
    if max_tick is None or max_tick <= 0:
        return None
    return int(max_tick)


def snapshot_inventory(session: Session, run_id: str) -> SnapshotInventory:
    ticks = list_snapshot_ticks(run_id)
    expected = expected_tick_count(session, run_id)
    if not ticks:
        status = "unavailable"
    elif expected is None:
        status = "partial"
    elif ticks == list(range(1, expected + 1)):
        status = "complete"
    else:
        status = "partial"
    return SnapshotInventory(
        status=status,
        count=len(ticks),
        expected_ticks=expected,
        ticks=ticks,
        latest_tick=max(ticks) if ticks else None,
    )


def metric_delta_rows(run_id: str) -> list[dict[str, Any]]:
    snapshots = load_snapshots(run_id)
    rows = []
    previous: dict[str, Any] | None = None
    for snapshot in snapshots:
        metrics = snapshot.get("metrics", {})
        row: dict[str, Any] = {
            "tick": snapshot.get("tick"),
            "metrics": metrics,
            "deltas": {},
            "household_resource_summary": snapshot.get("household_resource_summary"),
            "upkeep_summary": snapshot.get("upkeep_summary"),
            "propagation_summary": snapshot.get("propagation_summary"),
        }
        for key in DELTA_METRICS:
            current_value = metrics.get(key)
            previous_value = previous.get(key) if previous else None
            row["deltas"][key] = _delta(current_value, previous_value)
        rows.append(row)
        previous = metrics
    return rows


def _agents_summary(agents: list[Agent]) -> dict[str, Any]:
    by_archetype: dict[str, int] = {}
    by_status: dict[str, int] = {}
    average_needs = {"food": 0.0, "rest": 0.0, "belonging": 0.0}
    for agent in agents:
        by_archetype[agent.archetype] = by_archetype.get(agent.archetype, 0) + 1
        by_status[agent.status] = by_status.get(agent.status, 0) + 1
        for need in average_needs:
            average_needs[need] += float(agent.needs.get(need, 0))
    if agents:
        average_needs = {key: round(value / len(agents), 2) for key, value in average_needs.items()}
    return {
        "count": len(agents),
        "by_archetype": dict(sorted(by_archetype.items())),
        "by_status": dict(sorted(by_status.items())),
        "average_needs": average_needs,
    }


def _households_summary(households: list[Household]) -> dict[str, Any]:
    sizes = [len(household.members) for household in households]
    return {
        "count": len(households),
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "households": [
            {
                "household_id": household.household_id,
                "household_name": household.household_name,
                "kinship_type": household.kinship_type,
                "member_count": len(household.members),
            }
            for household in households
        ],
    }


def _relationships_summary(relationships: list[Relationship]) -> dict[str, Any]:
    count = len(relationships)
    avg_affinity = sum(relationship.affinity for relationship in relationships) / count if count else 0
    avg_trust = sum(relationship.trust for relationship in relationships) / count if count else 0
    avg_reputation = sum(relationship.reputation for relationship in relationships) / count if count else 0
    return {
        "count": count,
        "average_affinity": round(avg_affinity, 3),
        "average_trust": round(avg_trust, 3),
        "average_reputation": round(avg_reputation, 3),
    }


def _resources_summary(resources: list[Resource]) -> dict[str, Any]:
    totals: dict[str, float] = {}
    by_owner: dict[str, dict[str, float]] = {}
    for resource in resources:
        totals[resource.resource_type] = totals.get(resource.resource_type, 0.0) + resource.quantity
        owner = by_owner.setdefault(resource.owner_id, {})
        owner[resource.resource_type] = round(resource.quantity, 2)
    return {
        "totals": {key: round(value, 2) for key, value in sorted(totals.items())},
        "by_owner": dict(sorted(by_owner.items())),
    }


def _delta(current_value: Any, previous_value: Any) -> float | None:
    if current_value is None or previous_value is None:
        return None
    try:
        return round(float(current_value) - float(previous_value), 5)
    except (TypeError, ValueError):
        return None
