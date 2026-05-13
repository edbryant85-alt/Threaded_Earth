from __future__ import annotations

import json

from typer.testing import CliRunner

from threaded_earth.cli import app
from threaded_earth.config import DiagnosticsConfig, load_config
from threaded_earth.db import session_factory
from threaded_earth.diagnostics import diagnose_run
from threaded_earth.models import Event, Household, Relationship, Resource, Run
from threaded_earth.paths import diagnostics_json_path, diagnostics_report_path, snapshot_path
from threaded_earth.simulation import initialize_run, run_simulation
from threaded_earth.web import run_detail


def test_resource_runaway_collapse_flatline_and_saturation_detection(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    SessionLocal = session_factory(tmp_path / "run-test" / "threaded_earth.sqlite")
    with SessionLocal() as session:
        _seed_minimal_run(session)
        _write_snapshot("run-test", 1, total_food=10, stress=0.5)
        _write_snapshot("run-test", 2, total_food=30, stress=0.5)
        for index in range(4):
            session.add(
                Relationship(
                    run_id="run-test",
                    source_agent=f"a{index}",
                    target_agent=f"b{index}",
                    affinity=0.95,
                    trust=0.96,
                    reputation=0.97,
                    kinship_relation="settlement tie",
                )
            )
        result = diagnose_run(session, "run-test", DiagnosticsConfig(flatline_window_ticks=2))

    metrics = {warning["metric_name"]: warning for warning in result["warnings"]}
    assert "total_food" in metrics
    assert "trust" in metrics
    assert "reputation" in metrics
    assert "resource_stress" in metrics


def test_resource_collapse_and_dead_subsystem_detection(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    SessionLocal = session_factory(tmp_path / "run-test" / "threaded_earth.sqlite")
    with SessionLocal() as session:
        _seed_minimal_run(session)
        _write_snapshot("run-test", 1, total_food=30, stress=0.2)
        _write_snapshot("run-test", 2, total_food=5, stress=0.2)
        result = diagnose_run(session, "run-test", DiagnosticsConfig(flatline_window_ticks=2))

    severities = {warning["metric_name"]: warning["severity"] for warning in result["warnings"]}
    assert severities["total_food"] == "critical"
    assert "resource_exchange" in severities
    assert "social_propagation" in severities


def test_diagnostics_command_creates_json_markdown_and_dashboard_renders(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.cli.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.web.ARTIFACTS_DIR", tmp_path)
    runner = CliRunner()
    config = load_config()
    SessionLocal = session_factory(tmp_path / "run-test" / "threaded_earth.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 42, config)
        run_simulation(session, "run-test", 2, 42, config)

    result = runner.invoke(app, ["diagnose", "--run-id", "run-test"])

    assert result.exit_code == 0, result.output
    assert diagnostics_json_path("run-test").exists()
    assert diagnostics_report_path("run-test").exists()
    data = json.loads(diagnostics_json_path("run-test").read_text(encoding="utf-8"))
    assert data["run_id"] == "run-test"
    assert all("severity" in warning for warning in data["warnings"])
    html = run_detail("run-test")
    assert "Stability Diagnostics" in html


def test_diagnostics_output_is_deterministic_for_fixed_data(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    SessionLocal = session_factory(tmp_path / "run-test" / "threaded_earth.sqlite")
    with SessionLocal() as session:
        _seed_minimal_run(session)
        _write_snapshot("run-test", 1, total_food=10, stress=0.4)
        _write_snapshot("run-test", 2, total_food=30, stress=0.4)
        first = diagnose_run(session, "run-test", DiagnosticsConfig(flatline_window_ticks=2))
        second = diagnose_run(session, "run-test", DiagnosticsConfig(flatline_window_ticks=2))

    assert first == second


def _seed_minimal_run(session) -> None:
    session.add(Run(run_id="run-test", seed=1, config={"simulation": {"population_size": 1}}, status="complete"))
    session.add(
        Household(
            household_id="hh-001",
            run_id="run-test",
            household_name="Ala Hearth",
            kinship_type="extended",
            members=[],
            stored_resources={"grain": 1, "fish": 1, "reeds": 1, "tools": 1},
        )
    )
    for resource_type, quantity in {"grain": 1, "fish": 1, "reeds": 1, "tools": 1}.items():
        session.add(Resource(run_id="run-test", resource_type=resource_type, owner_scope="household", owner_id="hh-001", quantity=quantity))
    session.add(Event(event_id="run-test-ev-000001", run_id="run-test", tick=1, event_type="rest", actor=None, target=None, payload={}, summary="rest"))
    session.flush()


def _write_snapshot(run_id: str, tick: int, total_food: float, stress: float) -> None:
    path = snapshot_path(run_id, tick)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "tick": tick,
                "household_resource_summary": {"total_food": total_food, "total_materials": 4},
                "upkeep_summary": {"households_with_shortage": 0},
                "propagation_summary": {"propagation_events_this_tick": 0},
                "metrics": {
                    "resource_stress": stress,
                    "relationship_density": 0.1,
                    "reputation_variance": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
