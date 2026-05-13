from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from threaded_earth.goals import goal_stats
from threaded_earth.metrics import compute_metrics, write_metrics
from threaded_earth.models import Decision, Event, Goal, Memory, Resource, Run
from threaded_earth.paths import report_path
from threaded_earth.resources import household_resource_summary
from threaded_earth.snapshots import DELTA_METRICS, metric_delta_rows
from threaded_earth.targeting import SOCIAL_ACTIONS, target_stats


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
    target_lines = _targeted_social_lines(session, run_id)
    household_resource_lines = _household_resource_lines(session, run_id)

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
                "## Targeted Social Actions",
                *target_lines,
                "",
                "## Household Resources",
                *household_resource_lines,
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


def _targeted_social_lines(session: Session, run_id: str) -> list[str]:
    stats = target_stats(session, run_id)
    decisions = (
        session.query(Decision)
        .filter(Decision.run_id == run_id)
        .order_by(Decision.tick, Decision.agent_id)
        .all()
    )
    targeted = [
        decision
        for decision in decisions
        if decision.selected_action.get("action") in SOCIAL_ACTIONS and decision.selected_target_agent_id
    ]
    untargeted = [
        decision
        for decision in decisions
        if decision.selected_action.get("action") in SOCIAL_ACTIONS and not decision.selected_target_agent_id
    ]
    most_targeted = ", ".join(
        f"{agent_id}={count}" for agent_id, count in stats["most_targeted_agents"].items()
    ) or "none"
    examples = [
        f"- example tick {decision.tick} {decision.agent_id} -> {decision.selected_target_agent_id}: "
        f"{decision.selected_action.get('action')} because {', '.join(decision.target_selection_reasons[:4])}"
        for decision in targeted
    ][:5]
    return [
        f"- targeted social decisions: {stats['targeted_social_decisions']}",
        f"- untargeted social decisions: {stats['untargeted_social_decisions']}",
        f"- most-targeted agents: {most_targeted}",
        *(examples or ["- examples: no targeted social decisions recorded."]),
        *([f"- warning: {len(untargeted)} social decisions had no target."] if untargeted else []),
    ]


def _household_resource_lines(session: Session, run_id: str) -> list[str]:
    summary = household_resource_summary(session, run_id)
    transfer_events = [
        event
        for event in session.query(Event).filter(Event.run_id == run_id).order_by(Event.tick, Event.event_id).all()
        if event.payload.get("resource_transfer")
    ]
    notable_transfers = []
    for event in transfer_events[:8]:
        transfer = event.payload.get("resource_transfer") or {}
        notable_transfers.append(
            f"- tick {event.tick} {event.event_type}: {transfer.get('status')} "
            f"{transfer.get('transferred_quantity')} {transfer.get('resource_type')} "
            f"from {transfer.get('source_household_id')} to {transfer.get('target_household_id')}"
        )
    rows = metric_delta_rows(run_id)
    resource_changes = []
    upkeep_lines = []
    repeated_shortages: dict[str, int] = {}
    previous_food = None
    previous_materials = None
    for row in rows:
        snapshot_summary = row.get("household_resource_summary")
        if not snapshot_summary:
            continue
        food = snapshot_summary.get("total_food")
        materials = snapshot_summary.get("total_materials")
        food_delta = "n/a" if previous_food is None else f"{food - previous_food:+.2f}"
        material_delta = "n/a" if previous_materials is None else f"{materials - previous_materials:+.2f}"
        resource_changes.append(f"- tick {row['tick']}: food={food} ({food_delta}); materials={materials} ({material_delta})")
        upkeep = row.get("upkeep_summary") or {}
        if upkeep:
            upkeep_lines.append(
                f"- tick {row['tick']}: consumed={upkeep.get('food_consumed_this_tick', 0)}; "
                f"shortage={upkeep.get('total_shortage_amount', 0)}; "
                f"households_with_shortage={upkeep.get('households_with_shortage', 0)}"
            )
            for household_id in upkeep.get("shortage_household_ids", []):
                repeated_shortages[household_id] = repeated_shortages.get(household_id, 0) + 1
        previous_food = food
        previous_materials = materials
    influenced = [
        decision
        for decision in session.query(Decision).filter(Decision.run_id == run_id).order_by(Decision.tick, Decision.agent_id).all()
        if any("household_food=" in reason or "household_materials=" in reason for reason in decision.reasons)
    ][:3]
    examples = [
        f"- decision tick {decision.tick} {decision.agent_id}: {decision.selected_action.get('action')} because {', '.join(decision.reasons[:5])}"
        for decision in influenced
    ]
    return [
        f"- total food: {summary['total_food']}",
        f"- total materials: {summary['total_materials']}",
        f"- average food per household: {summary['average_food_per_household']}",
        f"- households under scarcity threshold: {summary['households_below_scarcity_threshold']}",
        f"- repeated shortage households: {', '.join(f'{household}={count}' for household, count in sorted(repeated_shortages.items())) or 'none'}",
        *(upkeep_lines[:10] or ["- consumption by tick: no upkeep snapshots available."]),
        *(notable_transfers or ["- notable transfers: none recorded."]),
        *(resource_changes[:8] or ["- resource changes by tick: no snapshot resource summaries available."]),
        *(examples or ["- examples: no resource-influenced decisions found."]),
    ]
