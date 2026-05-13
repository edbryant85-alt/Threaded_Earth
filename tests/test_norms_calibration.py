from __future__ import annotations

import json
import random

from threaded_earth.cognition import choose_action
from threaded_earth.config import RoleConfig, load_config
from threaded_earth.db import session_factory
from threaded_earth.events import record_event
from threaded_earth.models import Agent, Event, Household, NormCandidate, Relationship, RoleSignal, Run
from threaded_earth.norms import update_norm_candidate, update_norm_candidates_for_event
from threaded_earth.reports import generate_report
from threaded_earth.simulation import initialize_run, run_simulation
from threaded_earth.snapshots import snapshot_path
from threaded_earth.web import run_detail


def test_low_score_roles_do_not_influence_decisions():
    role = _role("role-low", "provider", 0.2)
    trace = choose_action(_agent("agent-001"), _household(), [], random.Random(1), 1, active_roles=[role])

    assert trace.role_signals_seen == ["role-low"]
    assert trace.role_signals_applied == []
    assert trace.role_score_adjustments == {}
    assert trace.role_adjustment_total == 0.0


def test_irrelevant_roles_do_not_influence_unrelated_actions():
    role = _role("role-trader", "trader", 0.9)
    trace = choose_action(_agent("agent-001"), _household(), [], random.Random(1), 1, active_roles=[role])
    repair = next(candidate for candidate in trace.candidate_actions if candidate["action"] == "repair_relationship")

    assert trace.role_signals_applied == ["role-trader"]
    assert trace.role_score_adjustments == {"share_food": 0.06}
    assert repair["role_adjustment"] == 0.0


def test_total_role_adjustment_is_capped_and_diagnostics_are_consistent():
    roles = [_role("role-provider", "provider", 1.0), _role("role-helper", "helper", 1.0), _role("role-mediator", "mediator", 1.0)]
    config = RoleConfig(role_influence_max_adjustment=0.04, role_influence_min_score=0.45)
    trace = choose_action(_agent("agent-001"), _household(), [], random.Random(1), 1, active_roles=roles, role_config=config)

    assert trace.role_adjustment_total == 0.04
    assert trace.role_adjustment_capped is True
    assert len(trace.role_signals_applied) <= len(trace.role_signals_seen)
    assert sum(abs(value) for value in trace.role_score_adjustments.values()) <= 0.041


def test_role_influence_is_deterministic():
    roles = [_role("role-helper", "helper", 0.8)]
    first = choose_action(_agent("agent-001"), _household(), [], random.Random(7), 1, active_roles=roles)
    second = choose_action(_agent("agent-001"), _household(), [], random.Random(7), 1, active_roles=roles)

    assert first.selected_action == second.selected_action
    assert first.role_score_adjustments == second.role_score_adjustments
    assert first.candidate_actions == second.candidate_actions


def test_repeated_sharing_with_kin_creates_stable_sharing_norm(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        actor, target, relationship = _seed_norm_fixture(session)
        for tick in range(1, 5):
            event = _event(session, tick, "resource_exchange", actor.neutral_id, target.neutral_id, transferred=1.0)
            update_norm_candidates_for_event(session, "run-test", event, actor, target, relationship, tick)
        norm = _norm(session, "sharing_food_with_kin")

    assert norm.evidence_count == 4
    assert norm.support_score >= 1.2
    assert norm.status == "stable"


def test_repeated_repairs_and_conflicts_update_norm_candidates(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        actor, target, relationship = _seed_norm_fixture(session)
        for tick in range(1, 4):
            repair = _event(session, tick, "repair", actor.neutral_id, target.neutral_id)
            update_norm_candidates_for_event(session, "run-test", repair, actor, target, relationship, tick)
        for tick in range(4, 7):
            conflict = _event(session, tick, "conflict", actor.neutral_id, target.neutral_id)
            update_norm_candidates_for_event(session, "run-test", conflict, actor, target, relationship, tick)

        repair_norm = _norm(session, "repairing_after_conflict")
        conflict_norm = _norm(session, "disfavoring_repeated_conflict")

    assert repair_norm.evidence_count == 6
    assert repair_norm.support_score > repair_norm.opposition_score
    assert conflict_norm.evidence_count == 3
    assert conflict_norm.support_score > 0


def test_contradictory_behavior_increases_norm_opposition(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        actor, target, relationship = _seed_norm_fixture(session)
        success = _event(session, 1, "resource_exchange", actor.neutral_id, target.neutral_id, transferred=1.0)
        failure = _event(session, 2, "resource_exchange", actor.neutral_id, target.neutral_id, transferred=0.0)
        update_norm_candidates_for_event(session, "run-test", success, actor, target, relationship, 1)
        update_norm_candidates_for_event(session, "run-test", failure, actor, target, relationship, 2)
        norm = _norm(session, "sharing_food_with_kin")

    assert norm.support_score > 0
    assert norm.opposition_score > 0
    assert norm.status == "emerging"


def test_norm_candidates_persist_and_status_updates_deterministically(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        _seed_norm_fixture(session)
        first = update_norm_candidate(session, "run-test", "valuing_providers", 0.3, 0.0, 1, "provider helped")
        second = update_norm_candidate(session, "run-test", "valuing_providers", 0.3, 0.0, 2, "provider helped again")
        session.commit()
        norm_id = second.norm_candidate_id

    with SessionLocal() as session:
        norm = session.get(NormCandidate, norm_id)

    assert first.norm_candidate_id == norm_id
    assert norm is not None
    assert norm.evidence_count == 2
    assert norm.first_observed_tick == 1
    assert norm.last_observed_tick == 2


def test_snapshots_reports_and_dashboard_include_norm_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.web.ARTIFACTS_DIR", tmp_path)
    config = load_config()
    SessionLocal = session_factory(tmp_path / "run-test" / "threaded_earth.sqlite")
    with SessionLocal() as session:
        initialize_run(session, "run-test", 51, config)
        run_simulation(session, "run-test", 3, 51, config)
        report_path = generate_report(session, "run-test")

    snapshot = json.loads(snapshot_path("run-test", 3).read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    html = run_detail("run-test")
    assert "norm_candidates_total" in snapshot["norm_summary"]
    assert "## Norm Candidates" in report
    assert "descriptive candidates" in report
    assert "Norm Candidates" in html


def test_default_norm_influence_is_diagnostic_only():
    config = load_config()
    assert config.simulation.norms.norm_influence_enabled is False


def _seed_norm_fixture(session):
    session.add(Run(run_id="run-test", seed=1, config={}, status="running"))
    actor = _agent("agent-001")
    target = _agent("agent-002")
    session.add_all([actor, target, _household()])
    relationship = Relationship(
        run_id="run-test",
        source_agent=actor.neutral_id,
        target_agent=target.neutral_id,
        affinity=0.65,
        trust=0.7,
        reputation=0.55,
        kinship_relation="household kin",
    )
    session.add(relationship)
    session.flush()
    return actor, target, relationship


def _agent(agent_id: str) -> Agent:
    return Agent(
        neutral_id=agent_id,
        run_id="run-test",
        display_name=agent_id,
        archetype="cultivator",
        age_band="adult",
        household_id="hh-001",
        traits={},
        needs={"food": 55, "rest": 55, "belonging": 55},
        status="active",
    )


def _household() -> Household:
    return Household(
        household_id="hh-001",
        run_id="run-test",
        household_name="Ala Hearth",
        kinship_type="extended",
        members=["agent-001", "agent-002"],
        stored_resources={"grain": 20, "fish": 2, "reeds": 2, "tools": 1},
    )


def _role(role_id: str, role_name: str, score: float) -> RoleSignal:
    return RoleSignal(
        role_signal_id=role_id,
        run_id="run-test",
        agent_id="agent-001",
        role_name=role_name,
        score=score,
        evidence_count=1,
        created_tick=1,
        updated_tick=1,
        evidence_summary="test role",
    )


def _event(session, tick: int, event_type: str, actor: str, target: str, transferred: float = 0.0) -> Event:
    payload = {}
    if event_type == "resource_exchange":
        payload = {"resource_transfer": {"transferred_quantity": transferred}}
    return record_event(session, "run-test", tick, event_type, actor, target, payload, event_type)


def _norm(session, norm_name: str) -> NormCandidate:
    return session.query(NormCandidate).filter(NormCandidate.run_id == "run-test", NormCandidate.norm_name == norm_name).one()
