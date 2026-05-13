from __future__ import annotations

from typer.testing import CliRunner

from threaded_earth.cli import app
from threaded_earth.config import load_config
from threaded_earth.db import session_factory
from threaded_earth.events import event_log_path
from threaded_earth.models import Agent, Decision, Event, Household, Run
from threaded_earth.paths import report_path
from threaded_earth.reports import generate_report
from threaded_earth.simulation import initialize_run, run_simulation


def test_db_initialization_and_generation(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.events.event_log_path", lambda run_id: tmp_path / run_id / "logs" / "events.jsonl")
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 7, config)
        assert session.get(Run, "run-test") is not None
        assert session.query(Agent).count() == 50
        assert session.query(Household).count() >= 9
        assert session.query(Event).filter(Event.event_type == "run_initialized").count() == 1


def test_simulation_tick_event_logging_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.events.event_log_path", lambda run_id: tmp_path / run_id / "logs" / "events.jsonl")
    monkeypatch.setattr("threaded_earth.paths.event_log_path", lambda run_id: tmp_path / run_id / "logs" / "events.jsonl")
    monkeypatch.setattr("threaded_earth.paths.report_path", lambda run_id: tmp_path / run_id / "reports" / "report.md")
    monkeypatch.setattr("threaded_earth.reports.report_path", lambda run_id: tmp_path / run_id / "reports" / "report.md")
    monkeypatch.setattr("threaded_earth.metrics.metrics_path", lambda run_id: tmp_path / run_id / "exports" / "metrics.json")
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 7, config)
        run_simulation(session, "run-test", 1, 7, config)
        path = generate_report(session, "run-test")
        assert session.query(Decision).count() == 50
        assert session.query(Event).count() > 1
        assert (tmp_path / "run-test" / "logs" / "events.jsonl").exists()
        assert path.exists()
        assert "Notable Decisions" in path.read_text(encoding="utf-8")


def test_cli_init():
    runner = CliRunner()
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "Initialized Threaded Earth workspace" in result.output


def test_cli_run_report_replay(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.cli.ARTIFACTS_DIR", tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--days", "1", "--seed", "11"])
    assert result.exit_code == 0, result.output
    run_id = [line for line in result.output.splitlines() if line.startswith("run_id=")][0].split("=", 1)[1]
    assert (tmp_path / run_id / "threaded_earth.sqlite").exists()
    assert event_log_path(run_id).exists()
    assert report_path(run_id).exists()

    report_result = runner.invoke(app, ["report", "--run-id", run_id])
    assert report_result.exit_code == 0, report_result.output
    replay_result = runner.invoke(app, ["replay", "--run-id", run_id])
    assert replay_result.exit_code == 0, replay_result.output
    assert "run_initialized" in replay_result.output
