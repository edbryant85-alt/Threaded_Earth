from __future__ import annotations

import json

from typer.testing import CliRunner

from threaded_earth.cli import app
from threaded_earth.config import load_config
from threaded_earth.db import session_factory
from threaded_earth.models import Decision
from threaded_earth.paths import snapshot_path
from threaded_earth.reports import generate_report
from threaded_earth.simulation import initialize_run, run_simulation
from threaded_earth.snapshots import snapshot_inventory
from threaded_earth.web import run_detail


def test_snapshot_creation_path_and_json_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 19, config)
        run_simulation(session, "run-test", 2, 19, config)

        first_snapshot = snapshot_path("run-test", 1)
        assert first_snapshot == tmp_path / "run-test" / "snapshots" / "tick_1.json"
        assert first_snapshot.exists()

        data = json.loads(first_snapshot.read_text(encoding="utf-8"))
        assert data["run_id"] == "run-test"
        assert data["tick"] == 1
        assert data["created_at"]
        assert data["agents_summary"]["count"] == 50
        assert data["households_summary"]["count"] >= 9
        assert data["relationships_summary"]["count"] > 0
        assert "totals" in data["resources_summary"]
        assert "total_memories" in data["memory_summary"]
        assert "total_active_goals" in data["goal_summary"]
        assert "targeted_social_decisions" in data["target_summary"]
        assert "decisions_with_target_aware_scores" in data["target_aware_summary"]
        assert "total_food" in data["household_resource_summary"]
        assert "resource_transfers_this_tick" in data
        assert "upkeep_summary" in data
        assert "food_consumed_this_tick" in data["upkeep_summary"]
        assert "relationship_density" in data["metrics"]
        assert len(data["event_ids"]) > 0
        assert session.query(Decision).count() == 100


def test_snapshot_inventory_complete_partial_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 23, config)
        run_simulation(session, "run-test", 3, 23, config)
        complete = snapshot_inventory(session, "run-test")
        assert complete.status == "complete"
        assert complete.count == 3
        assert complete.expected_ticks == 3
        assert complete.latest_tick == 3

        snapshot_path("run-test", 2).unlink()
        partial = snapshot_inventory(session, "run-test")
        assert partial.status == "partial"
        assert partial.count == 2

        snapshot_path("run-test", 1).unlink()
        snapshot_path("run-test", 3).unlink()
        unavailable = snapshot_inventory(session, "run-test")
        assert unavailable.status == "unavailable"
        assert unavailable.count == 0


def test_replay_reports_snapshot_statuses(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.cli.ARTIFACTS_DIR", tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--days", "2", "--seed", "29"])
    assert result.exit_code == 0, result.output
    run_id = [line for line in result.output.splitlines() if line.startswith("run_id=")][0].split("=", 1)[1]

    complete = runner.invoke(app, ["replay", "--run-id", run_id])
    assert complete.exit_code == 0, complete.output
    assert "snapshot_replay=complete" in complete.output
    assert "snapshots=2" in complete.output
    assert "expected_ticks=2" in complete.output

    snapshot_path(run_id, 2).unlink()
    partial = runner.invoke(app, ["replay", "--run-id", run_id])
    assert "snapshot_replay=partial" in partial.output

    snapshot_path(run_id, 1).unlink()
    unavailable = runner.invoke(app, ["replay", "--run-id", run_id])
    assert "snapshot_replay=unavailable" in unavailable.output


def test_report_includes_metric_deltas(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 31, config)
        run_simulation(session, "run-test", 2, 31, config)
        path = generate_report(session, "run-test")

    text = path.read_text(encoding="utf-8")
    assert "## Tick Metric Deltas" in text
    assert "## Memory Influence" in text
    assert "## Goal Dynamics" in text
    assert "## Targeted Social Actions" in text
    assert "target-aware" in text
    assert "## Household Resources" in text
    assert "shortage" in text
    assert "relationship_density" in text
    assert "conflict_frequency" in text
    assert "resource_stress" in text


def test_dashboard_run_page_renders_snapshot_information(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.web.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "run-test" / "threaded_earth.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 37, config)
        run_simulation(session, "run-test", 2, 37, config)

    html = run_detail("run-test")
    assert "Snapshots" in html
    assert "Replay data: <strong>complete</strong>" in html
    assert "Per-Tick Metrics" in html
    assert "Memory Observability" in html
    assert "Goal Observability" in html
    assert "Target Observability" in html
    assert "Target-aware decisions" in html
    assert "Household Resources" in html
    assert "Latest daily consumption" in html
    assert "Latest tick: <strong>2</strong>" in html
