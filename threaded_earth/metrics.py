from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.models import Event, Relationship, Resource
from threaded_earth.paths import metrics_path


def compute_metrics(session: Session, run_id: str) -> dict[str, Any]:
    relationships = session.query(Relationship).filter(Relationship.run_id == run_id).all()
    events = session.query(Event).filter(Event.run_id == run_id).all()
    resources = session.query(Resource).filter(Resource.run_id == run_id).all()
    reputations = [relationship.reputation for relationship in relationships]
    total_food = sum(resource.quantity for resource in resources if resource.resource_type in {"grain", "fish"})
    household_count = len({resource.owner_id for resource in resources if resource.owner_scope == "household"})
    avg_food = total_food / household_count if household_count else 0
    return {
        "relationship_density": round(len(relationships) / 50, 3),
        "conflict_frequency": sum(1 for event in events if event.event_type == "conflict"),
        "cooperation_frequency": sum(1 for event in events if event.event_type == "cooperation"),
        "reputation_variance": round(statistics.pvariance(reputations), 5) if len(reputations) > 1 else 0.0,
        "resource_stress": round(max(0.0, 1 - (avg_food / 24)), 3),
        "total_food": round(total_food, 2),
    }


def write_metrics(session: Session, run_id: str) -> Path:
    path = metrics_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compute_metrics(session, run_id), indent=2, sort_keys=True), encoding="utf-8")
    return path
