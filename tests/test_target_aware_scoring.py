from __future__ import annotations

import random

from threaded_earth.cognition import choose_action
from threaded_earth.memory import RetrievedMemory
from threaded_earth.models import Agent, Goal, Household, Relationship


def test_target_aware_scoring_evaluates_targets_before_final_action():
    trace = choose_action(
        _agent(),
        _household("hh-001", grain=20, fish=6),
        [_relationship("agent-002", trust=0.8, affinity=0.7, kinship="household kin")],
        random.Random(3),
        1,
        [],
        [],
        {"agent-001": _household("hh-001", 20, 6), "agent-002": _household("hh-001", 20, 6)},
    )
    cooperate = _candidate(trace, "cooperate")
    assert cooperate["target_aware_adjustment"] > 0
    assert cooperate["best_target"]["target_agent_id"] == "agent-002"
    assert "cooperate" in trace.best_target_by_action


def test_high_trust_kin_target_increases_cooperation_score():
    trace = choose_action(
        _agent(),
        _household("hh-001", 20, 6),
        [_relationship("agent-002", trust=0.82, affinity=0.74, kinship="household kin")],
        random.Random(3),
        1,
    )
    assert _candidate(trace, "cooperate")["target_aware_adjustment"] > 0.05


def test_actor_surplus_and_target_scarcity_increases_share_score():
    trace = choose_action(
        _agent(),
        _household("hh-001", 36, 8),
        [_relationship("agent-002", trust=0.72, affinity=0.68, kinship="settlement tie")],
        random.Random(3),
        1,
        [],
        [_goal("maintain_household", 0.8)],
        {"agent-001": _household("hh-001", 36, 8), "agent-002": _household("hh-002", 2, 1)},
    )
    share = _candidate(trace, "share_food")
    assert share["target_aware_adjustment"] > 0.05
    assert share["best_target"]["resource_bonus"] > 0


def test_damaged_relationship_increases_repair_score():
    trace = choose_action(
        _agent(),
        _household("hh-001", 20, 6),
        [_relationship("agent-002", trust=0.34, affinity=0.38, reputation=0.55)],
        random.Random(3),
        1,
        [],
        [_goal("repair_relationship", 0.9)],
    )
    assert _candidate(trace, "repair_relationship")["target_aware_adjustment"] > 0


def test_hostile_recent_conflict_target_increases_avoid_score():
    trace = choose_action(
        _agent(),
        _household("hh-001", 20, 6),
        [_relationship("agent-002", trust=0.2, affinity=0.25, reputation=0.35)],
        random.Random(3),
        3,
        [_memory("mem-conflict", "negative", ["agent-001", "agent-002"])],
        [_goal("avoid_conflict", 0.85)],
    )
    avoid = _candidate(trace, "avoid_conflict")
    assert avoid["target_aware_adjustment"] > 0
    assert avoid["best_target"]["memory_factor"]["negative"] > 0


def test_non_social_actions_do_not_require_target_aware_targets():
    trace = choose_action(_agent(), _household("hh-001", 1, 0), [], random.Random(1), 1)
    seek = _candidate(trace, "seek_food")
    assert seek.get("best_target") is None
    assert seek["target_aware_adjustment"] == 0.0


def _candidate(trace, action: str):
    return next(candidate for candidate in trace.candidate_actions if candidate["action"] == action)


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


def _household(household_id: str, grain: float, fish: float) -> Household:
    return Household(
        household_id=household_id,
        run_id="run-test",
        household_name=f"{household_id} Hearth",
        kinship_type="extended",
        members=["agent-001"],
        stored_resources={"grain": grain, "fish": fish, "reeds": 3, "tools": 2},
    )


def _relationship(
    target: str,
    trust: float,
    affinity: float,
    reputation: float = 0.6,
    kinship: str = "settlement tie",
) -> Relationship:
    return Relationship(
        run_id="run-test",
        source_agent="agent-001",
        target_agent=target,
        affinity=affinity,
        trust=trust,
        reputation=reputation,
        kinship_relation=kinship,
    )


def _goal(goal_type: str, priority: float) -> Goal:
    return Goal(
        goal_id=f"goal-{goal_type}",
        run_id="run-test",
        agent_id="agent-001",
        goal_type=goal_type,
        priority=priority,
        status="active",
        created_tick=1,
        updated_tick=1,
        source_reason="test",
        progress=0.0,
        notes="",
    )


def _memory(memory_id: str, polarity: str, agents: list[str]) -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=memory_id,
        event_id=f"{memory_id}-event",
        event_type="conflict",
        score=1.0,
        salience=0.8,
        recency=0.5,
        polarity=polarity,
        involved_agents=agents,
        summary=f"{polarity} memory",
    )
