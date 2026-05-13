from __future__ import annotations

from pathlib import Path
import json

from sqlalchemy.orm import Session

from threaded_earth.goals import goal_stats
from threaded_earth.metrics import compute_metrics, write_metrics
from threaded_earth.models import Decision, Event, Goal, Memory, NormCandidate, Resource, RoleSignal, Run
from threaded_earth.norms import norm_stats
from threaded_earth.paths import diagnostics_json_path, report_path
from threaded_earth.propagation import propagation_pressure_rows, propagation_stats
from threaded_earth.resources import household_resource_summary
from threaded_earth.roles import role_stats
from threaded_earth.snapshots import DELTA_METRICS, metric_delta_rows
from threaded_earth.targeting import SOCIAL_ACTIONS, target_aware_stats, target_stats


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
    propagation_lines = _social_propagation_lines(session, run_id)
    propagation_pressure_lines = _propagation_pressure_lines(session, run_id)
    role_lines = _role_stabilization_lines(session, run_id)
    norm_lines = _norm_candidate_lines(session, run_id)
    diagnostics_lines = _diagnostics_lines(run_id)

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
                "## Social Propagation",
                *propagation_lines,
                "",
                "## Propagation Pressure",
                *propagation_pressure_lines,
                "",
                "## Role Stabilization",
                *role_lines,
                "",
                "## Norm Candidates",
                *norm_lines,
                "",
                "## Stability Diagnostics",
                *diagnostics_lines,
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
    aware_stats = target_aware_stats(session, run_id)
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
    aware_examples = [
        f"- target-aware tick {decision.tick} {decision.agent_id}: {decision.selected_action.get('action')} "
        f"target={decision.selected_target_agent_id}; {', '.join(decision.target_aware_score_reasons[:3])}"
        for decision in targeted
        if any(decision.target_aware_action_scores.values())
    ][:5]
    reason_counts = ", ".join(
        f"{reason}={count}" for reason, count in aware_stats["common_target_aware_reasons"].items()
    ) or "none"
    return [
        f"- targeted social decisions: {stats['targeted_social_decisions']}",
        f"- untargeted social decisions: {stats['untargeted_social_decisions']}",
        f"- decisions with target-aware score contribution: {aware_stats['decisions_with_target_aware_scores']}",
        f"- social target candidates evaluated: {aware_stats['social_candidates_evaluated']}",
        f"- common target-aware reasons: {reason_counts}",
        f"- most-targeted agents: {most_targeted}",
        *(examples or ["- examples: no targeted social decisions recorded."]),
        *(aware_examples or ["- target-aware examples: no target-aware score contributions recorded."]),
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


def _social_propagation_lines(session: Session, run_id: str) -> list[str]:
    stats = propagation_stats(session, run_id)
    propagation_events = (
        session.query(Event)
        .filter(Event.run_id == run_id, Event.event_type == "social_propagation")
        .order_by(Event.tick, Event.event_id)
        .all()
    )
    reasons = ", ".join(
        f"{reason}={count}" for reason, count in stats["common_propagation_reasons"].items()
    ) or "none"
    visible = ", ".join(
        f"{agent_id}={count}" for agent_id, count in stats["most_socially_visible_agents"].items()
    ) or "none"
    examples = [
        f"- example tick {event.tick}: source={event.payload.get('source_event_id')} "
        f"observer={event.payload.get('observer_agent_id')} subject={event.payload.get('subject_agent_id')} "
        f"reason={event.payload.get('propagation_reason')} delta={event.payload.get('relationship_delta')}"
        for event in propagation_events[:5]
    ]
    return [
        f"- events propagated: {stats['propagated_event_count']}",
        f"- propagated memories: {stats['propagated_memory_count']}",
        f"- propagation skipped: {stats['propagation_skipped_count']}",
        f"- most common propagation reasons: {reasons}",
        f"- most socially visible agents: {visible}",
        *(examples or ["- examples: no social propagation events recorded."]),
    ]


def _propagation_pressure_lines(session: Session, run_id: str) -> list[str]:
    rows = propagation_pressure_rows(session, run_id)
    if not rows:
        return ["- no propagation pressure rows available."]
    total_skipped = sum(row["propagation_skipped_this_tick"] for row in rows)
    cap_ticks = [row["tick"] for row in rows if row["cap_reached"]]
    examples = [
        f"- tick {row['tick']}: events={row['propagation_events_this_tick']} "
        f"memories={row['propagation_memories_this_tick']} skipped={row['propagation_skipped_this_tick']} "
        f"cap_reached={row['cap_reached']}"
        for row in rows[:10]
    ]
    return [
        f"- total skipped propagation attempts: {total_skipped}",
        f"- ticks where cap reached: {', '.join(str(tick) for tick in cap_ticks) or 'none'}",
        *examples,
    ]


def _role_stabilization_lines(session: Session, run_id: str) -> list[str]:
    stats = role_stats(session, run_id)
    counts = ", ".join(
        f"{role_name}={count}" for role_name, count in stats["role_counts_above_threshold"].items()
    ) or "none"
    top_roles = [
        f"- top role {item['agent_id']}: {item['role_name']} score={item['score']} "
        f"evidence={item['evidence_count']} updated_tick={item['updated_tick']}"
        for item in stats["top_role_signals"][:6]
    ]
    influenced = (
        session.query(Decision)
        .filter(Decision.run_id == run_id)
        .order_by(Decision.tick, Decision.agent_id)
        .all()
    )
    influenced = [decision for decision in influenced if decision.role_signals_applied]
    examples = [
        f"- decision tick {decision.tick} {decision.agent_id}: {decision.selected_action.get('action')}; "
        f"{decision.role_influence_summary}"
        for decision in influenced[:5]
    ]
    recent_roles = (
        session.query(RoleSignal)
        .filter(RoleSignal.run_id == run_id)
        .order_by(RoleSignal.updated_tick.desc(), RoleSignal.score.desc(), RoleSignal.role_name)
        .limit(5)
        .all()
    )
    shifts = [
        f"- recent shift {role.agent_id}: {role.role_name} score={role.score:.2f}; {role.evidence_summary}"
        for role in recent_roles
    ]
    return [
        f"- role counts above threshold: {counts}",
        *(top_roles or ["- top role signals: none recorded."]),
        *(shifts or ["- notable role shifts: none recorded."]),
        *(examples or ["- example decisions influenced by roles: none recorded."]),
    ]


def _norm_candidate_lines(session: Session, run_id: str) -> list[str]:
    stats = norm_stats(session, run_id)
    norms = (
        session.query(NormCandidate)
        .filter(NormCandidate.run_id == run_id)
        .order_by(NormCandidate.support_score.desc(), NormCandidate.evidence_count.desc(), NormCandidate.norm_name)
        .all()
    )
    examples = [
        f"- {norm.norm_name}: status={norm.status}; evidence={norm.evidence_count}; "
        f"agents={len(norm.contributing_agent_ids or [])}; households={len(norm.contributing_household_ids or [])}; "
        f"breadth={norm.breadth_score:.2f}; density={norm.evidence_density:.2f}; "
        f"support={norm.support_score:.2f}; opposition={norm.opposition_score:.2f}; "
        f"last_tick={norm.last_observed_tick}; example={norm.evidence_summary}"
        for norm in norms[:8]
    ]
    return [
        "- These are descriptive candidates inferred from repeated logged patterns, not laws, institutions, or enforced rules.",
        "- Support is breadth-aware: repeated evidence from the same agent or household is discounted.",
        f"- total candidates: {stats['norm_candidates_total']}",
        f"- emerging/stable/declining: {stats['emerging_norms']}/{stats['stable_norms']}/{stats['declining_norms']}",
        *(examples or ["- detected norms: none yet."]),
    ]


def _diagnostics_lines(run_id: str) -> list[str]:
    path = diagnostics_json_path(run_id)
    if not path.exists():
        return ["- Diagnostics have not been generated for this run yet. Run `threaded-earth diagnose --run-id RUN_ID`."]
    data = json.loads(path.read_text(encoding="utf-8"))
    warnings = data.get("warnings", [])
    by_severity: dict[str, int] = {}
    for warning in warnings:
        by_severity[warning["severity"]] = by_severity.get(warning["severity"], 0) + 1
    lines = [f"- {severity}: {count}" for severity, count in sorted(by_severity.items())]
    lines.extend(
        f"- {warning['severity']} {warning['metric_name']}: {warning['evidence_text']}"
        for warning in warnings[:5]
    )
    return lines or ["- No diagnostics warnings recorded."]
