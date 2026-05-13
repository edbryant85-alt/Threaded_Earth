from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from threaded_earth.db import session_factory
from threaded_earth.models import Agent, Decision, Event, Run
from threaded_earth.paths import ARTIFACTS_DIR, db_path, metrics_path
from threaded_earth.snapshots import DELTA_METRICS, metric_delta_rows, snapshot_inventory


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
