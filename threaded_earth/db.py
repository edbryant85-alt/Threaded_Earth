from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from threaded_earth.models import Base


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path}"


def make_engine(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(sqlite_url(path), future=True)


def init_db(path: Path) -> None:
    engine = make_engine(path)
    Base.metadata.create_all(engine)


def session_factory(path: Path) -> sessionmaker[Session]:
    engine = make_engine(path)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session, future=True)
