from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from threaded_earth.goals import goal_stats
from threaded_earth.metrics import compute_metrics, write_metrics
from threaded_earth.models import Decision, Event, Goal, Memory, Resource, Run
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
    memory_lines = _memory_influence_lines(session, run_id)
    goal_lines = _goal_dynamics_lines(session, run_id)

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
                "## Memory Influence",
                *memory_lines,
                "",
                "## Goal Dynamics",
                *goal_lines,
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


def _memory_influence_lines(session: Session, run_id: str) -> list[str]:
    decisions = session.query(Decision).filter(Decision.run_id == run_id).order_by(Decision.tick, Decision.agent_id).all()
    influenced = [decision for decision in decisions if decision.retrieved_memory_ids]
    retrieved_ids = sorted({memory_id for decision in influenced for memory_id in decision.retrieved_memory_ids})
    event_types: dict[str, int] = {}
    if retrieved_ids:
        rows = (
            session.query(Memory, Event)
            .join(Event, Memory.event_id == Event.event_id)
            .filter(Memory.run_id == run_id, Memory.memory_id.in_(retrieved_ids))
            .all()
        )
        for _, event in rows:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    common_types = ", ".join(f"{event_type}={count}" for event_type, count in sorted(event_types.items())) or "none"
    examples = [
        f"- example tick {decision.tick} {decision.agent_id}: selected {decision.selected_action['action']}; "
        f"{decision.memory_influence_summary}"
        for decision in influenced
        if decision.memory_score_adjustments
    ][:3]
    return [
        f"- decisions influenced by memory: {len(influenced)} of {len(decisions)}",
        f"- most commonly retrieved memory types: {common_types}",
        *(examples or ["- examples: no decisions had non-zero memory score adjustments."]),
    ]


def _goal_dynamics_lines(session: Session, run_id: str) -> list[str]:
    stats = goal_stats(session, run_id)
    decisions = session.query(Decision).filter(Decision.run_id == run_id).order_by(Decision.tick, Decision.agent_id).all()
    influenced = [decision for decision in decisions if decision.active_goal_ids]
    goals = session.query(Goal).filter(Goal.run_id == run_id).order_by(Goal.priority.desc(), Goal.goal_id).all()
    by_type = ", ".join(f"{goal_type}={count}" for goal_type, count in stats["goals_by_type"].items()) or "none"
    created_count = len(goals)
    examples = [
        f"- example tick {decision.tick} {decision.agent_id}: selected {decision.selected_action['action']}; "
        f"{decision.goal_influence_summary}"
        for decision in influenced
        if decision.goal_score_adjustments
    ][:3]
    notable = [
        f"- high priority {goal.agent_id}: {goal.goal_type} priority={goal.priority:.2f} status={goal.status} reason={goal.source_reason}"
        for goal in goals
        if goal.status == "active"
    ][:5]
    return [
        f"- active goals by type: {by_type}",
        f"- goals created/satisfied/abandoned: {created_count}/{stats['satisfied_count']}/{stats['abandoned_count']}",
        f"- decisions influenced by goals: {len(influenced)} of {len(decisions)}",
        *(examples or ["- examples: no decisions had non-zero goal score adjustments."]),
        *(notable or ["- notable agents: no active high-priority goals."]),
    ]
