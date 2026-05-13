from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from threaded_earth.db import session_factory
from threaded_earth.goals import goal_stats
from threaded_earth.memory import memory_stats
from threaded_earth.models import Agent, Decision, Event, Goal, Memory, NormCandidate, RoleSignal, Run
from threaded_earth.norms import norm_stats
from threaded_earth.paths import ARTIFACTS_DIR, db_path, metrics_path
from threaded_earth.propagation import propagation_pressure_rows, propagation_stats, recent_propagation_events
from threaded_earth.resources import household_resource_summary, upkeep_stats_for_tick
from threaded_earth.roles import role_stats
from threaded_earth.snapshots import DELTA_METRICS, metric_delta_rows, snapshot_inventory
from threaded_earth.targeting import SOCIAL_ACTIONS, target_aware_stats, target_stats


app = FastAPI(title="Threaded Earth")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    runs = sorted([path.name for path in ARTIFACTS_DIR.glob("run-*") if db_path(path.name).exists()], reverse=True)
    items = "\n".join(f'<li><a href="/runs/{run_id}">{run_id}</a></li>' for run_id in runs)
    return _page("Threaded Earth Runs", f"<h1>Threaded Earth</h1><ul>{items or '<li>No runs yet.</li>'}</ul>")


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(run_id: str) -> str:
    path = db_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    SessionLocal = session_factory(path)
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        agents = session.query(Agent).filter(Agent.run_id == run_id).count()
        events = session.query(Event).filter(Event.run_id == run_id).order_by(Event.tick.desc()).limit(20).all()
        decisions = session.query(Decision).filter(Decision.run_id == run_id).order_by(Decision.tick.desc()).limit(10).all()
        inventory = snapshot_inventory(session, run_id)
        memory_summary = memory_stats(session, run_id)
        goal_summary = goal_stats(session, run_id)
        target_summary = target_stats(session, run_id)
        target_aware_summary = target_aware_stats(session, run_id)
        propagation_summary = propagation_stats(session, run_id)
        pressure_rows = propagation_pressure_rows(session, run_id)[-10:]
        role_summary = role_stats(session, run_id)
        norm_summary = norm_stats(session, run_id)
        resource_summary = household_resource_summary(session, run_id)
        recent_goals = (
            session.query(Goal)
            .filter(Goal.run_id == run_id, Goal.status == "active")
            .order_by(Goal.priority.desc(), Goal.updated_tick.desc(), Goal.goal_id)
            .limit(10)
            .all()
        )
        recent_memories = (
            session.query(Memory)
            .filter(Memory.run_id == run_id)
            .order_by(Memory.created_tick.desc(), Memory.memory_id.desc())
            .limit(8)
            .all()
        )
        influenced_decisions = (
            session.query(Decision)
            .filter(Decision.run_id == run_id)
            .all()
        )
        recent_targeted = [
            decision
            for decision in sorted(influenced_decisions, key=lambda item: (item.tick, item.agent_id), reverse=True)
            if decision.selected_action.get("action") in SOCIAL_ACTIONS and decision.selected_target_agent_id
        ][:10]
        recent_target_aware = [
            decision
            for decision in sorted(influenced_decisions, key=lambda item: (item.tick, item.agent_id), reverse=True)
            if any(decision.target_aware_action_scores.values())
        ][:10]
        recent_resource_events = [
            event
            for event in events
            if event.payload.get("resource_transfer") or event.event_type == "resource_change"
        ][:10]
        latest_upkeep = upkeep_stats_for_tick(session, run_id, inventory.latest_tick or 0)
        recent_shortages = [
            event
            for event in events
            if event.event_type == "household_shortage"
        ][:10]
        recent_propagations = recent_propagation_events(session, run_id, 10)
        recent_roles = (
            session.query(RoleSignal)
            .filter(RoleSignal.run_id == run_id)
            .order_by(RoleSignal.score.desc(), RoleSignal.updated_tick.desc(), RoleSignal.role_name)
            .limit(12)
            .all()
        )
        recent_norms = (
            session.query(NormCandidate)
            .filter(NormCandidate.run_id == run_id)
            .order_by(NormCandidate.support_score.desc(), NormCandidate.last_observed_tick.desc(), NormCandidate.norm_name)
            .limit(12)
            .all()
        )
        influenced_count = sum(1 for decision in influenced_decisions if decision.retrieved_memory_ids)
        goal_influenced_count = sum(1 for decision in influenced_decisions if decision.active_goal_ids)
        role_influenced_count = sum(1 for decision in influenced_decisions if decision.role_signals_applied)
    metrics = {}
    if metrics_path(run_id).exists():
        metrics = json.loads(metrics_path(run_id).read_text(encoding="utf-8"))
    event_rows = "".join(f"<tr><td>{event.tick}</td><td>{event.event_type}</td><td>{event.summary}</td></tr>" for event in events)
    decision_rows = "".join(
        f"<tr><td>{decision.tick}</td><td>{decision.agent_id}</td><td>{decision.selected_action['action']}</td><td>{decision.confidence}</td></tr>"
        for decision in decisions
    )
    metric_items = "".join(f"<li><strong>{key}</strong>: {value}</li>" for key, value in metrics.items())
    delta_rows = _metric_table_rows(run_id)
    memory_rows = "".join(
        f"<tr><td>{memory.created_tick}</td><td>{memory.agent_id}</td><td>{memory.salience:.2f}</td><td>{memory.summary}</td></tr>"
        for memory in recent_memories
    )
    goal_type_rows = "".join(
        f"<tr><td>{goal_type}</td><td>{count}</td></tr>"
        for goal_type, count in goal_summary["goals_by_type"].items()
    )
    goal_rows = "".join(
        f"<tr><td>{goal.agent_id}</td><td>{goal.goal_type}</td><td>{goal.priority:.2f}</td><td>{goal.updated_tick}</td><td>{goal.source_reason}</td></tr>"
        for goal in recent_goals
    )
    target_rows = "".join(
        f"<tr><td>{agent_id}</td><td>{count}</td></tr>"
        for agent_id, count in target_summary["most_targeted_agents"].items()
    )
    recent_target_rows = "".join(
        f"<tr><td>{decision.tick}</td><td>{decision.agent_id}</td><td>{decision.selected_action.get('action')}</td><td>{decision.selected_target_agent_id}</td><td>{'; '.join(decision.target_selection_reasons[:3])}</td></tr>"
        for decision in recent_targeted
    )
    aware_reason_rows = "".join(
        f"<tr><td>{reason}</td><td>{count}</td></tr>"
        for reason, count in target_aware_summary["common_target_aware_reasons"].items()
    )
    recent_aware_rows = "".join(
        f"<tr><td>{decision.tick}</td><td>{decision.agent_id}</td><td>{decision.selected_action.get('action')}</td><td>{decision.selected_target_agent_id or ''}</td><td>{'; '.join(decision.target_aware_score_reasons[:3])}</td></tr>"
        for decision in recent_target_aware
    )
    household_resource_rows = "".join(
        f"<tr><td>{household['household_id']}</td><td>{household['household_name']}</td><td>{household['member_count']}</td><td>{household['food']}</td><td>{household['food_per_member']}</td><td>{household['materials']}</td></tr>"
        for household in resource_summary["households"]
    )
    resource_event_rows = "".join(
        f"<tr><td>{event.tick}</td><td>{event.event_type}</td><td>{event.summary}</td></tr>"
        for event in recent_resource_events
    )
    shortage_rows = "".join(
        f"<tr><td>{event.tick}</td><td>{event.target}</td><td>{event.payload.get('shortage_amount')}</td><td>{event.summary}</td></tr>"
        for event in recent_shortages
    )
    propagation_reason_rows = "".join(
        f"<tr><td>{reason}</td><td>{count}</td></tr>"
        for reason, count in propagation_summary["common_propagation_reasons"].items()
    )
    visible_agent_rows = "".join(
        f"<tr><td>{agent_id}</td><td>{count}</td></tr>"
        for agent_id, count in propagation_summary["most_socially_visible_agents"].items()
    )
    propagation_rows = "".join(
        f"<tr><td>{event.tick}</td><td>{event.payload.get('source_event_id')}</td><td>{event.payload.get('observer_agent_id')}</td><td>{event.payload.get('subject_agent_id')}</td><td>{event.payload.get('propagation_reason')}</td></tr>"
        for event in recent_propagations
    )
    pressure_table_rows = "".join(
        f"<tr><td>{row['tick']}</td><td>{row['propagation_events_this_tick']}</td><td>{row['propagation_memories_this_tick']}</td><td>{row['propagation_skipped_this_tick']}</td><td>{row['cap_reached']}</td></tr>"
        for row in pressure_rows
    )
    role_count_rows = "".join(
        f"<tr><td>{role_name}</td><td>{count}</td></tr>"
        for role_name, count in role_summary["role_counts_above_threshold"].items()
    )
    role_rows = "".join(
        f"<tr><td>{role.agent_id}</td><td>{role.role_name}</td><td>{role.score:.2f}</td><td>{role.evidence_count}</td><td>{role.updated_tick}</td><td>{role.evidence_summary}</td></tr>"
        for role in recent_roles
    )
    norm_rows = "".join(
        f"<tr><td>{norm.norm_name}</td><td>{norm.status}</td><td>{norm.evidence_count}</td><td>{norm.support_score:.2f}</td><td>{norm.opposition_score:.2f}</td><td>{norm.last_observed_tick}</td><td>{norm.evidence_summary}</td></tr>"
        for norm in recent_norms
    )
    latest_tick = inventory.latest_tick if inventory.latest_tick is not None else "none"
    expected_ticks = inventory.expected_ticks if inventory.expected_ticks is not None else "unknown"
    return _page(
        run_id,
        f"""
        <h1>{run_id}</h1>
        <p>Status: <strong>{run.status}</strong> | Seed: <strong>{run.seed}</strong> | Agents: <strong>{agents}</strong></p>
        <h2>Snapshots</h2>
        <p>Replay data: <strong>{inventory.status}</strong> | Snapshots: <strong>{inventory.count}</strong> | Expected ticks: <strong>{expected_ticks}</strong> | Latest tick: <strong>{latest_tick}</strong></p>
        <h2>Metrics</h2><ul>{metric_items}</ul>
        <h2>Per-Tick Metrics</h2>
        <table><tr><th>Tick</th><th>Relationship Density</th><th>Conflict Frequency</th><th>Cooperation Frequency</th><th>Resource Stress</th><th>Reputation Variance</th></tr>{delta_rows}</table>
        <h2>Memory Observability</h2>
        <p>Total memories: <strong>{memory_summary['total_memories']}</strong> | Average per agent: <strong>{memory_summary['average_memories_per_agent']}</strong> | High salience: <strong>{memory_summary['high_salience_memory_count']}</strong> | Memory-influenced decisions: <strong>{influenced_count}</strong></p>
        <table><tr><th>Tick</th><th>Agent</th><th>Salience</th><th>Memory</th></tr>{memory_rows or '<tr><td colspan="4">No memories recorded.</td></tr>'}</table>
        <h2>Goal Observability</h2>
        <p>Active goals: <strong>{goal_summary['total_active_goals']}</strong> | Average per agent: <strong>{goal_summary['average_active_goals_per_agent']}</strong> | Satisfied: <strong>{goal_summary['satisfied_count']}</strong> | Abandoned: <strong>{goal_summary['abandoned_count']}</strong> | Goal-influenced decisions: <strong>{goal_influenced_count}</strong></p>
        <table><tr><th>Goal Type</th><th>Active Count</th></tr>{goal_type_rows or '<tr><td colspan="2">No active goals.</td></tr>'}</table>
        <table><tr><th>Agent</th><th>Goal</th><th>Priority</th><th>Updated Tick</th><th>Reason</th></tr>{goal_rows or '<tr><td colspan="5">No active goals.</td></tr>'}</table>
        <h2>Target Observability</h2>
        <p>Targeted social decisions: <strong>{target_summary['targeted_social_decisions']}</strong> | Untargeted social decisions: <strong>{target_summary['untargeted_social_decisions']}</strong></p>
        <p>Target-aware decisions: <strong>{target_aware_summary['decisions_with_target_aware_scores']}</strong> | Social candidates evaluated: <strong>{target_aware_summary['social_candidates_evaluated']}</strong></p>
        <table><tr><th>Target Agent</th><th>Count</th></tr>{target_rows or '<tr><td colspan="2">No targeted social decisions.</td></tr>'}</table>
        <table><tr><th>Target-Aware Reason</th><th>Count</th></tr>{aware_reason_rows or '<tr><td colspan="2">No target-aware reasons.</td></tr>'}</table>
        <table><tr><th>Tick</th><th>Actor</th><th>Action</th><th>Target</th><th>Reasons</th></tr>{recent_target_rows or '<tr><td colspan="5">No recent targeted social actions.</td></tr>'}</table>
        <table><tr><th>Tick</th><th>Actor</th><th>Selected</th><th>Target</th><th>Target-Aware Reasons</th></tr>{recent_aware_rows or '<tr><td colspan="5">No recent target-aware decisions.</td></tr>'}</table>
        <h2>Household Resources</h2>
        <p>Total food: <strong>{resource_summary['total_food']}</strong> | Total materials: <strong>{resource_summary['total_materials']}</strong> | Average food: <strong>{resource_summary['average_food_per_household']}</strong> | Scarce households: <strong>{resource_summary['households_below_scarcity_threshold']}</strong></p>
        <p>Latest daily consumption: <strong>{latest_upkeep['food_consumed_this_tick']}</strong> | Latest shortage: <strong>{latest_upkeep['total_shortage_amount']}</strong> | Households with shortage: <strong>{latest_upkeep['households_with_shortage']}</strong></p>
        <table><tr><th>Household</th><th>Name</th><th>Members</th><th>Food</th><th>Food / Member</th><th>Materials</th></tr>{household_resource_rows}</table>
        <table><tr><th>Tick</th><th>Household</th><th>Shortage</th><th>Shortage Event</th></tr>{shortage_rows or '<tr><td colspan="4">No recent shortages.</td></tr>'}</table>
        <table><tr><th>Tick</th><th>Type</th><th>Resource Event</th></tr>{resource_event_rows or '<tr><td colspan="3">No recent resource events.</td></tr>'}</table>
        <h2>Social Propagation</h2>
        <p>Propagated events: <strong>{propagation_summary['propagated_event_count']}</strong> | Propagated memories: <strong>{propagation_summary['propagated_memory_count']}</strong> | Skipped propagation: <strong>{propagation_summary['propagation_skipped_count']}</strong></p>
        <table><tr><th>Propagation Reason</th><th>Count</th></tr>{propagation_reason_rows or '<tr><td colspan="2">No propagation reasons recorded.</td></tr>'}</table>
        <table><tr><th>Visible Agent</th><th>Count</th></tr>{visible_agent_rows or '<tr><td colspan="2">No socially visible agents recorded.</td></tr>'}</table>
        <table><tr><th>Tick</th><th>Source Event</th><th>Observer</th><th>Subject</th><th>Reason</th></tr>{propagation_rows or '<tr><td colspan="5">No recent propagation events.</td></tr>'}</table>
        <h2>Propagation Pressure</h2>
        <table><tr><th>Tick</th><th>Events</th><th>Memories</th><th>Skipped</th><th>Cap Reached</th></tr>{pressure_table_rows or '<tr><td colspan="5">No propagation pressure rows.</td></tr>'}</table>
        <h2>Role Observability</h2>
        <p>Role-influenced decisions: <strong>{role_influenced_count}</strong></p>
        <table><tr><th>Role</th><th>Agents Above Threshold</th></tr>{role_count_rows or '<tr><td colspan="2">No role signals above threshold.</td></tr>'}</table>
        <table><tr><th>Agent</th><th>Role</th><th>Score</th><th>Evidence</th><th>Updated Tick</th><th>Evidence Summary</th></tr>{role_rows or '<tr><td colspan="6">No role signals recorded.</td></tr>'}</table>
        <h2>Norm Candidates</h2>
        <p>Total: <strong>{norm_summary['norm_candidates_total']}</strong> | Emerging: <strong>{norm_summary['emerging_norms']}</strong> | Stable: <strong>{norm_summary['stable_norms']}</strong> | Declining: <strong>{norm_summary['declining_norms']}</strong></p>
        <table><tr><th>Norm</th><th>Status</th><th>Evidence</th><th>Support</th><th>Opposition</th><th>Last Tick</th><th>Evidence Summary</th></tr>{norm_rows or '<tr><td colspan="7">No norm candidates recorded.</td></tr>'}</table>
        <h2>Recent Events</h2><table><tr><th>Tick</th><th>Type</th><th>Summary</th></tr>{event_rows}</table>
        <h2>Recent Decisions</h2><table><tr><th>Tick</th><th>Agent</th><th>Selected</th><th>Confidence</th></tr>{decision_rows}</table>
        """,
    )


def _page(title: str, body: str) -> str:
    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
      <style>
        body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; color: #1d2428; background: #f7f5ef; }}
        h1, h2 {{ color: #20352f; }}
        table {{ border-collapse: collapse; width: 100%; background: white; }}
        th, td {{ border: 1px solid #ccd3cc; padding: 0.45rem; text-align: left; vertical-align: top; }}
        th {{ background: #e6ece4; }}
        a {{ color: #265f73; }}
      </style>
    </head>
    <body>{body}</body>
    </html>
    """


def _metric_table_rows(run_id: str) -> str:
    rows = metric_delta_rows(run_id)
    if not rows:
        return '<tr><td colspan="6">No snapshots available.</td></tr>'
    html_rows = []
    for row in rows:
        cells = [f"<td>{row['tick']}</td>"]
        for key in DELTA_METRICS:
            value = row["metrics"].get(key, "unavailable")
            delta = row["deltas"].get(key)
            delta_text = "n/a" if delta is None else f"{delta:+g}"
            cells.append(f"<td>{value} <small>({delta_text})</small></td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(html_rows)
