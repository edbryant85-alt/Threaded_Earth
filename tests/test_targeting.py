from __future__ import annotations

from threaded_earth.memory import RetrievedMemory
from threaded_earth.models import Agent, Goal, Household, Relationship
from threaded_earth.targeting import select_target_for_action


def test_non_social_action_has_no_target():
    selection = select_target_for_action("seek_food", _agent(), [_relationship("agent-002", 0.9, 0.9, "household kin")], [], [])
    assert selection.selected_target_agent_id is None
    assert selection.target_selection_candidates == []


def test_cooperation_prefers_high_trust_kin_positive_memory():
    relationships = [
        _relationship("agent-002", 0.8, 0.75, "household kin"),
        _relationship("agent-003", 0.35, 0.35, "settlement tie"),
    ]
    memories = [_memory("mem-1", "positive", ["agent-001", "agent-002"])]
    selection = select_target_for_action("cooperate", _agent(), relationships, memories, [])
    assert selection.selected_target_agent_id == "agent-002"
    assert selection.target_memory_factors["agent-002"]["positive"] > 0


def test_avoidance_prefers_low_trust_recent_conflict_target():
    relationships = [
        _relationship("agent-002", 0.75, 0.75, "household kin"),
        _relationship("agent-003", 0.18, 0.25, "settlement tie"),
    ]
    memories = [_memory("mem-2", "negative", ["agent-001", "agent-003"])]
    goal = _goal("avoid_conflict", 0.8)
    selection = select_target_for_action("avoid_conflict", _agent(), relationships, memories, [goal])
    assert selection.selected_target_agent_id == "agent-003"
    assert selection.target_memory_factors["agent-003"]["negative"] > 0
    assert "avoid_conflict" in selection.target_goal_factors["agent-003"]["goal_types"]


def test_repair_prefers_damaged_relationship_above_hostility_threshold():
    relationships = [
        _relationship("agent-002", 0.05, 0.05, "settlement tie"),
        _relationship("agent-003", 0.36, 0.4, "settlement tie"),
        _relationship("agent-004", 0.9, 0.9, "household kin"),
    ]
    selection = select_target_for_action("repair_relationship", _agent(), relationships, [], [_goal("repair_relationship", 0.9)])
    assert selection.selected_target_agent_id == "agent-003"


def test_target_selection_is_deterministic():
    relationships = [
        _relationship("agent-003", 0.5, 0.5, "settlement tie"),
        _relationship("agent-002", 0.5, 0.5, "settlement tie"),
    ]
    first = select_target_for_action("cooperate", _agent(), relationships, [], [])
    second = select_target_for_action("cooperate", _agent(), relationships, [], [])
    assert first.selected_target_agent_id == second.selected_target_agent_id == "agent-002"


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


def _relationship(target: str, trust: float, affinity: float, kinship: str) -> Relationship:
    return Relationship(
        run_id="run-test",
        source_agent="agent-001",
        target_agent=target,
        affinity=affinity,
        trust=trust,
        reputation=0.55,
        kinship_relation=kinship,
    )


def _memory(memory_id: str, polarity: str, agents: list[str]) -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=memory_id,
        event_id=f"{memory_id}-event",
        event_type="conflict" if polarity == "negative" else "cooperation",
        score=1.0,
        salience=0.8,
        recency=0.5,
        polarity=polarity,
        involved_agents=agents,
        summary=f"{polarity} memory",
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
