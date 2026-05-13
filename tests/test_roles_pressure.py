from __future__ import annotations

import json
import random

from threaded_earth.cognition import choose_action
from threaded_earth.config import load_config
from threaded_earth.db import session_factory
from threaded_earth.models import Agent, Decision, Event, Household, Relationship, RoleSignal, Run
from threaded_earth.reports import generate_report
from threaded_earth.roles import record_action_role_evidence, role_stats, update_role_signal
from threaded_earth.simulation import initialize_run, run_simulation
from threaded_earth.snapshots import snapshot_path
from threaded_earth.targeting import evaluate_target_aware_actions
from threaded_earth.web import run_detail


def test_propagation_caps_limit_events_and_memories_per_tick(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    config.simulation.propagation.propagation_max_events_per_tick = 3
    config.simulation.propagation.propagation_max_memories_per_tick = 2
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 42, config)
        run_simulation(session, "run-test", 1, 42, config)

        events = session.query(Event).filter(Event.run_id == "run-test", Event.tick == 1, Event.event_type == "social_propagation").all()
        memories = [
            event
            for event in events
            if event.payload.get("memory_id")
        ]
        skipped = session.query(Event).filter(Event.run_id == "run-test", Event.tick == 1, Event.event_type == "propagation_skipped").all()
        snapshot = json.loads(snapshot_path("run-test", 1).read_text(encoding="utf-8"))

    assert len(events) <= 3
    assert len(memories) <= 2
    assert sum(event.payload["skipped_count"] for event in skipped) > 0
    assert snapshot["propagation_summary"]["propagation_skipped_this_tick"] > 0
    assert snapshot["propagation_summary"]["cap_reached"] is True


def test_role_scores_increase_persist_and_are_not_exclusive(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        _seed_role_fixture(session)
        agent = session.get(Agent, "agent-001")
        event = Event(
            event_id="run-test-ev-000001",
            run_id="run-test",
            tick=1,
            event_type="resource_change",
            actor="agent-001",
            target="hh-001",
            payload={},
            summary="agent gathered grain",
        )
        session.add(event)
        record_action_role_evidence(session, "run-test", agent, 1, "seek_food", event)
        update_role_signal(session, "run-test", "agent-001", "helper", 0.3, 2, "manual cooperation evidence")
        provider = _role(session, "agent-001", "provider")
        helper = _role(session, "agent-001", "helper")

    assert provider.score > 0
    assert provider.updated_tick == 1
    assert helper.score > 0
    assert provider.role_signal_id != helper.role_signal_id


def test_role_influence_appears_in_decision_logs(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 43, config)
        run_simulation(session, "run-test", 2, 43, config)
        decision = (
            session.query(Decision)
            .filter(Decision.run_id == "run-test")
            .order_by(Decision.tick, Decision.agent_id)
            .first()
        )

    assert decision is not None
    assert isinstance(decision.role_score_adjustments, dict)
    assert "role" in decision.role_influence_summary.lower()


def test_trusted_neighbor_weakly_affects_target_aware_scoring():
    agent = _agent("agent-001", "hh-001")
    target = _agent("agent-002", "hh-002")
    relationship = Relationship(
        run_id="run-test",
        source_agent=agent.neutral_id,
        target_agent=target.neutral_id,
        affinity=0.55,
        trust=0.55,
        reputation=0.55,
        kinship_relation="distant",
    )
    without_role = evaluate_target_aware_actions(["cooperate"], agent, [relationship], [], [], {})
    with_role = evaluate_target_aware_actions(
        ["cooperate"],
        agent,
        [relationship],
        [],
        [],
        {},
        {
            "agent-002": [
                RoleSignal(
                    role_signal_id="role-1",
                    run_id="run-test",
                    agent_id="agent-002",
                    role_name="trusted_neighbor",
                    score=0.8,
                    evidence_count=3,
                    created_tick=1,
                    updated_tick=2,
                    evidence_summary="high trust",
                )
            ]
        },
    )

    assert with_role.best_target_by_action["cooperate"]["target_score"] > without_role.best_target_by_action["cooperate"]["target_score"]
    assert any("target_role_trusted_neighbor" in reason for reason in with_role.target_aware_score_reasons)


def test_snapshots_reports_and_dashboard_include_pressure_and_roles(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.web.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    config.simulation.propagation.propagation_max_events_per_tick = 4
    SessionLocal = session_factory(tmp_path / "run-test" / "threaded_earth.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 44, config)
        run_simulation(session, "run-test", 2, 44, config)
        report_path = generate_report(session, "run-test")
        stats = role_stats(session, "run-test")

    snapshot = json.loads(snapshot_path("run-test", 2).read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    html = run_detail("run-test")
    assert "propagation_skipped_this_tick" in snapshot["propagation_summary"]
    assert "role_counts_above_threshold" in snapshot["role_summary"]
    assert stats["top_role_signals"]
    assert "## Propagation Pressure" in report
    assert "## Role Stabilization" in report
    assert "Propagation Pressure" in html
    assert "Role Observability" in html


def test_role_adjustments_influence_action_scores():
    agent = _agent("agent-001", "hh-001")
    household = Household(
        household_id="hh-001",
        run_id="run-test",
        household_name="Ala Hearth",
        kinship_type="extended",
        members=["agent-001"],
        stored_resources={"grain": 20, "fish": 2, "reeds": 2, "tools": 1},
    )
    role = RoleSignal(
        role_signal_id="role-1",
        run_id="run-test",
        agent_id="agent-001",
        role_name="mediator",
        score=0.9,
        evidence_count=5,
        created_tick=1,
        updated_tick=3,
        evidence_summary="repair pattern",
    )
    trace = choose_action(agent, household, [], random.Random(1), 3, active_roles=[role])
    repair = next(candidate for candidate in trace.candidate_actions if candidate["action"] == "repair_relationship")

    assert repair["role_adjustment"] > 0
    assert trace.role_score_adjustments["repair_relationship"] > 0


def _seed_role_fixture(session) -> None:
    session.add(Run(run_id="run-test", seed=1, config={}, status="running"))
    session.add(_agent("agent-001", "hh-001"))
    session.add(
        Household(
            household_id="hh-001",
            run_id="run-test",
            household_name="Ala Hearth",
            kinship_type="extended",
            members=["agent-001"],
            stored_resources={"grain": 10, "fish": 2, "reeds": 1, "tools": 1},
        )
    )
    session.flush()


def _agent(agent_id: str, household_id: str) -> Agent:
    return Agent(
        neutral_id=agent_id,
        run_id="run-test",
        display_name=agent_id,
        archetype="cultivator",
        age_band="adult",
        household_id=household_id,
        traits={},
        needs={"food": 55, "rest": 55, "belonging": 55},
        status="active",
    )


def _role(session, agent_id: str, role_name: str) -> RoleSignal:
    return (
        session.query(RoleSignal)
        .filter(RoleSignal.run_id == "run-test", RoleSignal.agent_id == agent_id, RoleSignal.role_name == role_name)
        .one()
    )
