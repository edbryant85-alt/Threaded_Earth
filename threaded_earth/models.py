from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    status: Mapped[str] = mapped_column(String, default="initialized")


class Agent(Base):
    __tablename__ = "agents"

    neutral_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    archetype: Mapped[str] = mapped_column(String, nullable=False)
    age_band: Mapped[str] = mapped_column(String, nullable=False)
    household_id: Mapped[str] = mapped_column(String, index=True)
    traits: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    needs: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String, default="active")


class Household(Base):
    __tablename__ = "households"

    household_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    household_name: Mapped[str] = mapped_column(String, nullable=False)
    kinship_type: Mapped[str] = mapped_column(String, nullable=False)
    members: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    stored_resources: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)


class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    source_agent: Mapped[str] = mapped_column(String, index=True)
    target_agent: Mapped[str] = mapped_column(String, index=True)
    affinity: Mapped[float] = mapped_column(Float, nullable=False)
    trust: Mapped[float] = mapped_column(Float, nullable=False)
    reputation: Mapped[float] = mapped_column(Float, nullable=False)
    kinship_relation: Mapped[str] = mapped_column(String, nullable=False)


class Memory(Base):
    __tablename__ = "memories"

    memory_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    agent_id: Mapped[str] = mapped_column(String, index=True)
    event_id: Mapped[str] = mapped_column(String, index=True)
    salience: Mapped[float] = mapped_column(Float, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    created_tick: Mapped[int] = mapped_column(Integer, nullable=False)


class Resource(Base):
    __tablename__ = "resources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    resource_type: Mapped[str] = mapped_column(String, nullable=False)
    owner_scope: Mapped[str] = mapped_column(String, nullable=False)
    owner_id: Mapped[str] = mapped_column(String, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)


class Location(Base):
    __tablename__ = "locations"

    location_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    symbolic_description: Mapped[str] = mapped_column(Text, nullable=False)


class Event(Base):
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    tick: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    actor: Mapped[str] = mapped_column(String, nullable=True)
    target: Mapped[str] = mapped_column(String, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    agent_id: Mapped[str] = mapped_column(String, index=True)
    tick: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_actions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    selected_action: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    retrieved_memory_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    memory_influence_summary: Mapped[str] = mapped_column(Text, default="No memories retrieved.")
    memory_score_adjustments: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty_notes: Mapped[str] = mapped_column(Text, nullable=False)
