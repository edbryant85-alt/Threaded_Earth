from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from threaded_earth.metrics import compute_metrics, write_metrics
from threaded_earth.models import Decision, Event, Resource, Run
from threaded_earth.paths import report_path
from threaded_earth.snapshots import DELTA_METRICS, metric_delta_rows


def generate_report(session: Session, run_id: str) -> Path:
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"Unknown run_id: {run_id}")
    events = session.query(Event).filter(Event.run_id == run_id).order_by(Event.tick, Event.event_id).all()
    decisions = (
        session.query(Decision).filter(Decision.run_id == run_id).order_by(Decision.tick, Decision.agent_id).limit(12).all()
    )
    resources = session.query(Resource).filter(Resource.run_id == run_id).order_by(Resource.owner_id).all()
    metrics = compute_metrics(session, run_id)
    write_metrics(session, run_id)

    major_events = [event for event in events if event.event_type in {"conflict", "cooperation", "resource_exchange"}][:20]
    resource_lines = [f"- {resource.owner_id}: {resource.resource_type}={resource.quantity:.2f}" for resource in resources[:30]]
    decision_lines = [
        f"- tick {decision.tick} {decision.agent_id}: selected {decision.selected_action['action']} "
        f"(confidence {decision.confidence}) because {', '.join(decision.reasons[:4])}"
        for decision in decisions
    ]
    event_lines = [f"- tick {event.tick} {event.event_type}: {event.summary}" for event in major_events]
    metric_lines = [f"- {key}: {value}" for key, value in metrics.items()]
    delta_lines = _metric_delta_lines(run_id)

    path = report_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# Threaded Earth Report: {run_id}",
                "",
                "## Run Settings",
                f"- seed: {run.seed}",
                f"- status: {run.status}",
                f"- population_size: {run.config['simulation']['population_size']}",
                f"- settlement: {run.config['simulation']['settlement']['name']}",
                "",
                "## Basic Metrics",
                *metric_lines,
                "",
                "## Tick Metric Deltas",
                *delta_lines,
                "",
                "## Major Events",
                *(event_lines or ["- No major events recorded."]),
                "",
                "## Notable Decisions",
                *decision_lines,
                "",
                "## Resource Changes",
                *resource_lines,
                "",
                "## Tensions And Conflicts",
                f"- conflict_frequency: {metrics['conflict_frequency']}",
                "- Harmful dynamics are logged explicitly; no suffering mechanics are hidden or amplified for spectacle.",
                "",
                "## Unresolved Dynamics",
                "- Household food pressure and changing trust are visible but intentionally simple.",
                "- Institutions, governance, religion, warfare, and multi-settlement dynamics are not implemented in this slice.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _metric_delta_lines(run_id: str) -> list[str]:
    rows = metric_delta_rows(run_id)
    if not rows:
        return ["- No per-tick snapshots available; metric deltas unavailable."]
    lines = []
    for row in rows:
        pieces = []
        for key in DELTA_METRICS:
            delta = row["deltas"].get(key)
            value = row["metrics"].get(key, "unavailable")
            delta_text = "n/a" if delta is None else f"{delta:+g}"
            pieces.append(f"{key}={value} ({delta_text})")
        lines.append(f"- tick {row['tick']}: " + "; ".join(pieces))
    return lines
