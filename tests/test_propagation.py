from __future__ import annotations

import json

import pytest

from threaded_earth.config import load_config
from threaded_earth.db import session_factory
from threaded_earth.events import record_event
from threaded_earth.models import Agent, Event, Household, Memory, Relationship, Run
from threaded_earth.propagation import propagate_social_event, propagation_stats
from threaded_earth.reports import generate_report
from threaded_earth.snapshots import write_snapshot
from threaded_earth.web import run_detail


def test_cooperation_trade_and_conflict_trigger_bounded_propagation(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    config.simulation.propagation.propagation_max_observers = 2
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        agents_by_id, households_by_agent = _seed_social_fixture(session, config)
        for event_type in ["cooperation", "resource_exchange", "conflict"]:
            event = record_event(session, "run-test", 1, event_type, "agent-001", "agent-002", {}, event_type)
            emitted = propagate_social_event(
                session,
                "run-test",
                1,
                event,
                agents_by_id,
                households_by_agent,
                config.simulation.propagation,
            )
            assert 0 < len(emitted) <= 2

        propagation_events = session.query(Event).filter(Event.event_type == "social_propagation").all()
        by_source: dict[str, int] = {}
        for event in propagation_events:
            by_source[event.payload["source_event_id"]] = by_source.get(event.payload["source_event_id"], 0) + 1
            assert event.payload["depth"] == 1
            assert event.payload["source_event_id"] not in {item.event_id for item in propagation_events}
        assert all(count <= 2 for count in by_source.values())


def test_propagation_is_deterministic_for_same_state(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    first = _propagation_signature(tmp_path / "first.sqlite")
    second = _propagation_signature(tmp_path / "second.sqlite")
    assert first == second


def test_propagated_memory_is_indirect_and_lower_salience_than_direct_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        agents_by_id, households_by_agent = _seed_social_fixture(session, config)
        event = record_event(session, "run-test", 1, "cooperation", "agent-001", "agent-002", {}, "actor helped target")
        session.add(
            Memory(
                memory_id="run-test-mem-000001",
                run_id="run-test",
                agent_id="agent-001",
                event_id=event.event_id,
                salience=0.58,
                summary=event.summary,
                created_tick=1,
            )
        )

        emitted = propagate_social_event(
            session,
            "run-test",
            1,
            event,
            agents_by_id,
            households_by_agent,
            config.simulation.propagation,
        )
        memory_id = emitted[0].payload["memory_id"]
        propagated = session.get(Memory, memory_id)

    assert propagated is not None
    assert propagated.salience < 0.58
    assert propagated.summary.startswith("Indirect social memory")


def test_propagated_relationship_effect_is_weaker_than_direct_effect(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        agents_by_id, households_by_agent = _seed_social_fixture(session, config)
        before = _relationship(session, "agent-003", "agent-001").trust
        event = record_event(session, "run-test", 1, "conflict", "agent-001", "agent-002", {}, "actor clashed with target")
        emitted = propagate_social_event(
            session,
            "run-test",
            1,
            event,
            agents_by_id,
            households_by_agent,
            config.simulation.propagation,
        )
        after = _relationship(session, "agent-003", "agent-001").trust
        delta = emitted[0].payload["relationship_delta"]["trust"]

    assert abs(after - before) == pytest.approx(abs(delta))
    assert abs(delta) < 0.06


def test_disabled_propagation_config_prevents_propagation(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    config.simulation.propagation.propagation_enabled = False
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        agents_by_id, households_by_agent = _seed_social_fixture(session, config)
        event = record_event(session, "run-test", 1, "cooperation", "agent-001", "agent-002", {}, "actor helped target")
        emitted = propagate_social_event(
            session,
            "run-test",
            1,
            event,
            agents_by_id,
            households_by_agent,
            config.simulation.propagation,
        )

    assert emitted == []


def test_snapshots_reports_and_dashboard_surface_propagation(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.web.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    db_path = tmp_path / "run-test" / "threaded_earth.sqlite"
    SessionLocal = session_factory(db_path)
    with SessionLocal() as session:
        agents_by_id, households_by_agent = _seed_social_fixture(session, config)
        event = record_event(session, "run-test", 1, "cooperation", "agent-001", "agent-002", {}, "actor helped target")
        propagate_social_event(
            session,
            "run-test",
            1,
            event,
            agents_by_id,
            households_by_agent,
            config.simulation.propagation,
        )
        session.commit()

        snapshot_path = write_snapshot(session, "run-test", 1)
        report_path = generate_report(session, "run-test")
        stats = propagation_stats(session, "run-test")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    html = run_detail("run-test")
    assert snapshot["propagation_summary"]["propagated_events_this_tick"] > 0
    assert stats["propagated_memory_count"] > 0
    assert "## Social Propagation" in report
    assert "Social Propagation" in html


def _propagation_signature(db_path):
    config = load_config()
    SessionLocal = session_factory(db_path)
    with SessionLocal() as session:
        agents_by_id, households_by_agent = _seed_social_fixture(session, config)
        event = record_event(session, "run-test", 1, "cooperation", "agent-001", "agent-002", {}, "actor helped target")
        emitted = propagate_social_event(
            session,
            "run-test",
            1,
            event,
            agents_by_id,
            households_by_agent,
            config.simulation.propagation,
        )
        return [
            (
                item.payload["observer_agent_id"],
                item.payload["subject_agent_id"],
                item.payload["propagation_reason"],
                item.payload["relationship_delta"],
            )
            for item in emitted
        ]


def _seed_social_fixture(session, config):
    session.add(Run(run_id="run-test", seed=42, config=config.as_dict(), status="running"))
    households = [
        Household(
            household_id="hh-001",
            run_id="run-test",
            household_name="Luma Hearth",
            kinship_type="extended",
            members=["agent-001", "agent-003", "agent-004"],
            stored_resources={"grain": 10, "fish": 2, "reeds": 1, "tools": 1},
        ),
        Household(
            household_id="hh-002",
            run_id="run-test",
            household_name="Navo Hearth",
            kinship_type="extended",
            members=["agent-002", "agent-005"],
            stored_resources={"grain": 8, "fish": 1, "reeds": 2, "tools": 1},
        ),
    ]
    agents = [
        _agent("agent-001", "Luma", "hh-001"),
        _agent("agent-002", "Navo", "hh-002"),
        _agent("agent-003", "Sira", "hh-001"),
        _agent("agent-004", "Talo", "hh-001"),
        _agent("agent-005", "Mira", "hh-002"),
    ]
    session.add_all(households + agents)
    for source, target in [
        ("agent-003", "agent-001"),
        ("agent-004", "agent-001"),
        ("agent-005", "agent-002"),
        ("agent-001", "agent-003"),
        ("agent-001", "agent-004"),
        ("agent-002", "agent-005"),
    ]:
        session.add(
            Relationship(
                run_id="run-test",
                source_agent=source,
                target_agent=target,
                affinity=0.68,
                trust=0.7,
                reputation=0.55,
                kinship_relation="household",
            )
        )
    session.flush()
    agents_by_id = {agent.neutral_id: agent for agent in agents}
    households_by_agent = {agent.neutral_id: session.get(Household, agent.household_id) for agent in agents}
    return agents_by_id, households_by_agent


def _agent(agent_id: str, name: str, household_id: str) -> Agent:
    return Agent(
        neutral_id=agent_id,
        run_id="run-test",
        display_name=name,
        archetype="cultivator",
        age_band="adult",
        household_id=household_id,
        traits={},
        needs={"food": 60, "rest": 60, "belonging": 60},
        status="active",
    )


def _relationship(session, source: str, target: str) -> Relationship:
    relationship = (
        session.query(Relationship)
        .filter(Relationship.run_id == "run-test", Relationship.source_agent == source, Relationship.target_agent == target)
        .one()
    )
    return relationship
