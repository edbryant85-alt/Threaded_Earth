from __future__ import annotations

from typer.testing import CliRunner

from threaded_earth.cli import app
from threaded_earth.config import load_config
from threaded_earth.db import session_factory
from threaded_earth.events import event_log_path
from threaded_earth.models import Agent, Decision, Event, Household, Memory, Relationship, Resource, Run
from threaded_earth.paths import report_path, snapshot_path
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
        decision = session.query(Decision).order_by(Decision.agent_id).first()
        assert decision is not None
        assert isinstance(decision.retrieved_memory_ids, list)
        assert isinstance(decision.memory_score_adjustments, dict)
        assert decision.memory_influence_summary
        assert isinstance(decision.active_goal_ids, list)
        assert isinstance(decision.goal_score_adjustments, dict)
        assert decision.goal_influence_summary
        assert isinstance(decision.target_selection_candidates, list)
        assert isinstance(decision.target_selection_scores, dict)
        assert isinstance(decision.target_aware_action_scores, dict)
        assert isinstance(decision.best_target_by_action, dict)
        assert isinstance(decision.target_aware_score_reasons, list)
        assert isinstance(decision.final_score_breakdown, dict)
        assert session.query(Event).count() > 1
        assert (tmp_path / "run-test" / "logs" / "events.jsonl").exists()
        assert (tmp_path / "run-test" / "snapshots" / "tick_1.json").exists()
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
    assert snapshot_path(run_id, 1).exists()

    report_result = runner.invoke(app, ["report", "--run-id", run_id])
    assert report_result.exit_code == 0, report_result.output
    replay_result = runner.invoke(app, ["replay", "--run-id", run_id])
    assert replay_result.exit_code == 0, replay_result.output
    assert "snapshot_replay=complete" in replay_result.output
    assert "run_initialized" in replay_result.output


def test_social_decisions_events_memories_and_relationships_use_selected_targets(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 43, config)
        run_simulation(session, "run-test", 3, 43, config)

        targeted = [
            decision
            for decision in session.query(Decision).filter(Decision.run_id == "run-test").all()
            if decision.selected_target_agent_id
        ]
        assert targeted
        sample = targeted[0]
        assert sample.selected_action["selected_target_agent_id"] == sample.selected_target_agent_id
        assert sample.target_selection_candidates
        assert sample.target_selection_reasons

        event = (
            session.query(Event)
            .filter(Event.run_id == "run-test", Event.actor == sample.agent_id, Event.target == sample.selected_target_agent_id)
            .first()
        )
        assert event is not None
        assert event.payload["selected_target_agent_id"] == sample.selected_target_agent_id

        actor_memory = session.query(Memory).filter(Memory.agent_id == sample.agent_id, Memory.event_id == event.event_id).first()
        target_memory = (
            session.query(Memory)
            .filter(Memory.agent_id == sample.selected_target_agent_id, Memory.event_id == event.event_id)
            .first()
        )
        assert actor_memory is not None
        assert target_memory is not None

        relationship = (
            session.query(Relationship)
            .filter(Relationship.source_agent == sample.agent_id, Relationship.target_agent == sample.selected_target_agent_id)
            .first()
        )
        assert relationship is not None


def test_resource_events_and_household_state_are_consistent(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 47, config)
        run_simulation(session, "run-test", 4, 47, config)

        resource_events = [
            event
            for event in session.query(Event).filter(Event.run_id == "run-test").all()
            if event.payload.get("resource_transfer") or event.event_type == "resource_change"
        ]
        assert resource_events
        assert any(event.payload.get("resource_transfer") for event in resource_events)

        for resource in session.query(Resource).filter(Resource.run_id == "run-test").all():
            household = session.get(Household, resource.owner_id)
            assert household is not None
            assert household.stored_resources[resource.resource_type] == resource.quantity
            assert resource.quantity >= 0


def test_conflict_remains_bounded_with_target_aware_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 61, config)
        run_simulation(session, "run-test", 3, 61, config)
        total_decisions = session.query(Decision).filter(Decision.run_id == "run-test").count()
        conflicts = session.query(Event).filter(Event.run_id == "run-test", Event.event_type == "conflict").count()

    assert conflicts / total_decisions < 0.15


def test_household_upkeep_shortage_events_and_needs(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    config.simulation.starting_resources.grain_per_household = 0
    config.simulation.starting_resources.fish_per_household = 0
    config.simulation.upkeep.daily_food_need_per_agent = 1.0
    config.simulation.upkeep.food_shortage_memory_threshold = 1.0
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 53, config)
        first_agent_before = session.get(Agent, "agent-001").needs["food"]
        run_simulation(session, "run-test", 1, 53, config)
        shortage = session.query(Event).filter(Event.run_id == "run-test", Event.event_type == "household_shortage").first()
        first_agent_after = session.get(Agent, "agent-001").needs["food"]
        memory = session.query(Memory).filter(Memory.agent_id == "agent-001").first()

    assert shortage is not None
    assert shortage.payload["shortage_amount"] > 0
    assert first_agent_after < first_agent_before
    assert memory is not None


def test_simulation_resource_path_is_deterministic_by_seed(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    summaries = []
    for index in range(2):
        SessionLocal = session_factory(tmp_path / f"test-{index}.sqlite")
        run_id = f"run-test-{index}"
        with SessionLocal() as session:
            initialize_run(session, run_id, 59, config)
            run_simulation(session, run_id, 2, 59, config)
            summaries.append(
                [
                    (event.tick, event.event_type, event.actor, event.target, event.summary)
                    for event in session.query(Event).filter(Event.run_id == run_id).order_by(Event.tick, Event.event_type, Event.summary).all()
                ]
            )

    assert summaries[0] == summaries[1]
