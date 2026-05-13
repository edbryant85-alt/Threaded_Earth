from __future__ import annotations

import random

from threaded_earth.cognition import choose_action
from threaded_earth.db import session_factory
from threaded_earth.models import Agent, Household, Resource, Run
from threaded_earth.resources import (
    add_household_resource,
    consume_household_food,
    household_food,
    household_resource_summary,
    transfer_household_resource,
)


def test_resource_seeking_increases_actor_household_resources(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        _seed_household(session, "hh-001", grain=4.0)
        delta = add_household_resource(session, "run-test", "hh-001", "grain", 2.5)
        household = session.get(Household, "hh-001")

    assert delta == 2.5
    assert household is not None
    assert household.stored_resources["grain"] == 6.5


def test_trade_share_transfers_between_households(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        _seed_household(session, "hh-001", grain=5.0)
        _seed_household(session, "hh-002", grain=1.0)
        transfer = transfer_household_resource(session, "run-test", "hh-001", "hh-002", "grain", 1.5)
        source = session.get(Household, "hh-001")
        target = session.get(Household, "hh-002")

    assert transfer["status"] == "success"
    assert transfer["transferred_quantity"] == 1.5
    assert source.stored_resources["grain"] == 3.5
    assert target.stored_resources["grain"] == 2.5


def test_transfer_cannot_create_negative_resources_and_records_limited_status(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        _seed_household(session, "hh-001", grain=0.4)
        _seed_household(session, "hh-002", grain=1.0)
        transfer = transfer_household_resource(session, "run-test", "hh-001", "hh-002", "grain", 1.5)
        source = session.get(Household, "hh-001")

    assert transfer["status"] == "limited"
    assert transfer["transferred_quantity"] == 0.4
    assert source.stored_resources["grain"] == 0.0


def test_household_food_decreases_by_member_count(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        _seed_household(session, "hh-001", grain=10.0, fish=2.0)
        household = session.get(Household, "hh-001")
        household.members = ["agent-001", "agent-002", "agent-003"]
        result = consume_household_food(session, "run-test", "hh-001", len(household.members) * 1.0)

    assert result["consumed_food"] == 3.0
    assert result["shortage_amount"] == 0.0
    assert household_food(household) == 9.0


def test_upkeep_shortage_never_makes_resources_negative(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        _seed_household(session, "hh-001", grain=0.25, fish=0.0)
        result = consume_household_food(session, "run-test", "hh-001", 3.0)
        household = session.get(Household, "hh-001")

    assert result["consumed_food"] == 0.25
    assert result["shortage_amount"] == 2.75
    assert household_food(household) == 0.0


def test_scarcity_increases_resource_seeking_score():
    scarce = choose_action(_agent(), _household(grain=2, fish=1), [], random.Random(1), 1)
    sufficient = choose_action(_agent(), _household(grain=24, fish=8), [], random.Random(1), 1)
    scarce_seek = _candidate(scarce, "seek_food")
    sufficient_seek = _candidate(sufficient, "seek_food")
    assert scarce_seek["base_score"] > sufficient_seek["base_score"]


def test_sufficient_resources_increase_share_score():
    scarce = choose_action(_agent(), _household(grain=3, fish=1), [], random.Random(1), 1)
    sufficient = choose_action(_agent(), _household(grain=30, fish=8), [], random.Random(1), 1)
    assert _candidate(sufficient, "share_food")["base_score"] > _candidate(scarce, "share_food")["base_score"]


def test_household_resource_summary_reports_scarcity(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        _seed_household(session, "hh-001", grain=4.0, fish=1.0, reeds=2.0, tools=1.0)
        _seed_household(session, "hh-002", grain=20.0, fish=5.0, reeds=4.0, tools=2.0)
        summary = household_resource_summary(session, "run-test")

    assert summary["total_food"] == 30.0
    assert summary["total_materials"] == 9.0
    assert summary["households_below_scarcity_threshold"] == 1


def _candidate(trace, action: str):
    return next(candidate for candidate in trace.candidate_actions if candidate["action"] == action)


def _seed_household(
    session,
    household_id: str,
    grain: float,
    fish: float = 0.0,
    reeds: float = 0.0,
    tools: float = 0.0,
) -> None:
    session.add(Run(run_id="run-test", seed=1, config={}, status="running")) if session.get(Run, "run-test") is None else None
    stored = {"grain": grain, "fish": fish, "reeds": reeds, "tools": tools}
    session.add(
        Household(
            household_id=household_id,
            run_id="run-test",
            household_name=f"{household_id} Hearth",
            kinship_type="extended",
            members=[],
            stored_resources=stored,
        )
    )
    for resource_type, quantity in stored.items():
        session.add(
            Resource(
                run_id="run-test",
                resource_type=resource_type,
                owner_scope="household",
                owner_id=household_id,
                quantity=quantity,
            )
        )
    session.flush()


def _agent() -> Agent:
    return Agent(
        neutral_id="agent-001",
        run_id="run-test",
        display_name="Alana",
        archetype="cultivator",
        age_band="adult",
        household_id="hh-001",
        traits={},
        needs={"food": 35, "rest": 30, "belonging": 35},
        status="active",
    )


def _household(grain: float, fish: float) -> Household:
    return Household(
        household_id="hh-001",
        run_id="run-test",
        household_name="Ala Hearth",
        kinship_type="extended",
        members=["agent-001"],
        stored_resources={"grain": grain, "fish": fish, "reeds": 2, "tools": 1},
    )
