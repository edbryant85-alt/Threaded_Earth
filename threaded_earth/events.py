from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.models import Event
from threaded_earth.paths import event_log_path


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def record_event(
    session: Session,
    run_id: str,
    tick: int,
    event_type: str,
    actor: str | None,
    target: str | None,
    payload: dict[str, Any],
    summary: str,
) -> Event:
    count = session.query(Event).filter(Event.run_id == run_id).count()
    event = Event(
        event_id=f"{run_id}-ev-{count + 1:06d}",
        run_id=run_id,
        tick=tick,
        event_type=event_type,
        actor=actor,
        target=target,
        payload=payload,
        summary=summary,
    )
    session.add(event)
    session.flush()
    append_jsonl(
        event_log_path(run_id),
        {
            "event_id": event.event_id,
            "run_id": run_id,
            "tick": tick,
            "event_type": event_type,
            "actor": actor,
            "target": target,
            "payload": payload,
            "summary": summary,
        },
    )
    return event
