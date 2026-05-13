from __future__ import annotations

import random

from threaded_earth.cognition import choose_action
from threaded_earth.db import session_factory
from threaded_earth.goals import update_agent_goals
from threaded_earth.memory import RetrievedMemory
from threaded_earth.models import Agent, Goal, Household, Relationship, Run


def test_goal_creation_from_needs_and_resource_stress(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        session.add(Run(run_id="run-test", seed=1, config={}, status="running"))
        agent = _agent(food=18, rest=30)
        household = _household(grain=8, fish=2)
        goals = update_agent_goals(session, "run-test", agent, household, [_relationship(trust=0.6)], [], 1)
        session.commit()

        goal_types = {goal.goal_type for goal in goals}
        assert "secure_food" in goal_types
        assert "maintain_household" in goal_types


def test_goal_creation_from_relationship_and_memory_conditions(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        session.add(Run(run_id="run-test", seed=1, config={}, status="running"))
        goals = update_agent_goals(
            session,
            "run-test",
            _agent(food=40, rest=30),
            _household(grain=18, fish=8),
            [_relationship(trust=0.25, affinity=0.25)],
            [_memory("mem-conflict", "conflict", "negative")],
            1,
        )

    goal_types = {goal.goal_type for goal in goals}
    assert "repair_relationship" in goal_types
    assert "avoid_conflict" in goal_types


def test_goals_persist_and_duplicate_active_goals_are_updated(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        session.add(Run(run_id="run-test", seed=1, config={}, status="running"))
        agent = _agent(food=18, rest=30)
        household = _household(grain=8, fish=2)
        update_agent_goals(session, "run-test", agent, household, [_relationship(trust=0.6)], [], 1)
        first_count = session.query(Goal).filter(Goal.run_id == "run-test").count()
        first_goal = session.query(Goal).filter(Goal.goal_type == "secure_food").one()

        agent.needs = {"food": 16, "rest": 30, "belonging": 35}
        update_agent_goals(session, "run-test", agent, household, [_relationship(trust=0.6)], [], 2)
        second_count = session.query(Goal).filter(Goal.run_id == "run-test").count()
        updated_goal = session.query(Goal).filter(Goal.goal_type == "secure_food").one()

    assert first_count == second_count
    assert updated_goal.goal_id == first_goal.goal_id
    assert updated_goal.updated_tick == 2
    assert updated_goal.status == "active"


def test_goals_influence_decision_scoring():
    goal = Goal(
        goal_id="goal-1",
        run_id="run-test",
        agent_id="agent-001",
        goal_type="secure_food",
        priority=0.9,
        status="active",
        created_tick=1,
        updated_tick=1,
        source_reason="test",
        progress=0.0,
        notes="",
    )
    trace = choose_action(
        _agent(food=34, rest=30),
        _household(grain=18, fish=5),
        [_relationship(trust=0.55)],
        random.Random(7),
        2,
        [],
        [goal],
    )
    seek_food = next(candidate for candidate in trace.candidate_actions if candidate["action"] == "seek_food")
    share_food = next(candidate for candidate in trace.candidate_actions if candidate["action"] == "share_food")
    assert seek_food["goal_adjustment"] > 0
    assert share_food["goal_adjustment"] < 0
    assert trace.active_goal_ids == ["goal-1"]
    assert trace.goal_score_adjustments


def _agent(food: int, rest: int) -> Agent:
    return Agent(
        neutral_id="agent-001",
        run_id="run-test",
        display_name="Alana",
        archetype="cultivator",
        age_band="adult",
        household_id="hh-001",
        traits={},
        needs={"food": food, "rest": rest, "belonging": 35},
        status="active",
    )


def _household(grain: float, fish: float) -> Household:
    return Household(
        household_id="hh-001",
        run_id="run-test",
        household_name="Ala Hearth",
        kinship_type="extended",
        members=["agent-001"],
        stored_resources={"grain": grain, "fish": fish},
    )


def _relationship(trust: float, affinity: float = 0.55) -> Relationship:
    return Relationship(
        run_id="run-test",
        source_agent="agent-001",
        target_agent="agent-002",
        affinity=affinity,
        trust=trust,
        reputation=0.5,
        kinship_relation="settlement tie",
    )


def _memory(memory_id: str, event_type: str, polarity: str) -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=memory_id,
        event_id=f"{memory_id}-event",
        event_type=event_type,
        score=1.0,
        salience=0.8,
        recency=0.5,
        polarity=polarity,
        involved_agents=["agent-001", "agent-002"],
        summary=f"{polarity} memory",
    )
