from __future__ import annotations

import random

from threaded_earth.cognition import choose_action
from threaded_earth.db import session_factory
from threaded_earth.memory import RetrievedMemory, retrieve_relevant_memories
from threaded_earth.models import Agent, Event, Household, Memory, Relationship, Run


def test_memory_retrieval_returns_relevant_memories_deterministically(tmp_path):
    SessionLocal = session_factory(tmp_path / "test.sqlite")
    with SessionLocal() as session:
        session.add(Run(run_id="run-test", seed=1, config={}, status="running"))
        session.add_all(
            [
                Event(
                    event_id="ev-1",
                    run_id="run-test",
                    tick=1,
                    event_type="cooperation",
                    actor="agent-001",
                    target="agent-002",
                    payload={},
                    summary="agent-001 cooperated with agent-002",
                ),
                Event(
                    event_id="ev-2",
                    run_id="run-test",
                    tick=1,
                    event_type="rest",
                    actor="agent-001",
                    target=None,
                    payload={},
                    summary="agent-001 rested",
                ),
            ]
        )
        session.add_all(
            [
                Memory(
                    memory_id="mem-1",
                    run_id="run-test",
                    agent_id="agent-001",
                    event_id="ev-1",
                    salience=0.8,
                    summary="cooperation with agent-002",
                    created_tick=1,
                ),
                Memory(
                    memory_id="mem-2",
                    run_id="run-test",
                    agent_id="agent-001",
                    event_id="ev-2",
                    salience=0.4,
                    summary="rested",
                    created_tick=1,
                ),
            ]
        )
        session.commit()

        relationships = [_relationship("agent-001", "agent-002", trust=0.55)]
        first = retrieve_relevant_memories(session, "run-test", "agent-001", 3, relationships)
        second = retrieve_relevant_memories(session, "run-test", "agent-001", 3, relationships)

    assert [memory.memory_id for memory in first] == ["mem-1", "mem-2"]
    assert [memory.memory_id for memory in first] == [memory.memory_id for memory in second]
    assert first[0].event_type == "cooperation"
    assert first[0].polarity == "positive"


def test_positive_memory_increases_cooperation_scoring():
    trace = choose_action(
        _agent(),
        _household(),
        [_relationship("agent-001", "agent-002", trust=0.55)],
        random.Random(7),
        5,
        [_retrieved_memory("mem-positive", "cooperation", "positive")],
    )
    assert _candidate(trace, "cooperate")["memory_adjustment"] > 0
    assert _candidate(trace, "share_food")["memory_adjustment"] > 0


def test_negative_memory_increases_conflict_scoring():
    trace = choose_action(
        _agent(),
        _household(),
        [_relationship("agent-001", "agent-002", trust=0.35)],
        random.Random(7),
        5,
        [_retrieved_memory("mem-negative", "conflict", "negative")],
    )
    assert _candidate(trace, "conflict_over_food")["memory_adjustment"] > 0
    assert _candidate(trace, "cooperate")["memory_adjustment"] < 0


def test_resource_stress_memory_increases_resource_seeking():
    trace = choose_action(
        _agent(),
        _household(),
        [_relationship("agent-001", "agent-002", trust=0.55)],
        random.Random(7),
        5,
        [_retrieved_memory("mem-stress", "resource_exchange", "resource_stress")],
    )
    assert _candidate(trace, "seek_food")["memory_adjustment"] > 0
    assert _candidate(trace, "share_food")["memory_adjustment"] < 0


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
        needs={"food": 32, "rest": 30, "belonging": 25},
        status="active",
    )


def _household() -> Household:
    return Household(
        household_id="hh-001",
        run_id="run-test",
        household_name="Ala Hearth",
        kinship_type="extended",
        members=["agent-001"],
        stored_resources={"grain": 18, "fish": 4},
    )


def _relationship(source: str, target: str, trust: float) -> Relationship:
    return Relationship(
        run_id="run-test",
        source_agent=source,
        target_agent=target,
        affinity=0.55,
        trust=trust,
        reputation=0.5,
        kinship_relation="settlement tie",
    )


def _retrieved_memory(memory_id: str, event_type: str, polarity: str) -> RetrievedMemory:
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
