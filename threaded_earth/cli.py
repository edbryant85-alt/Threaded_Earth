from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from threaded_earth.config import load_config
from threaded_earth.db import init_db, session_factory
from threaded_earth.events import append_jsonl
from threaded_earth.models import Event
from threaded_earth.paths import ARTIFACTS_DIR, DEFAULT_CONFIG_PATH, db_path, ensure_artifact_dirs, event_log_path, report_path
from threaded_earth.reports import generate_report
from threaded_earth.simulation import initialize_run, make_run_id, run_simulation


app = typer.Typer(help="Threaded Earth local civilization simulation tools.")


@app.command()
def init(config: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Config YAML path.")) -> None:
    loaded = load_config(config)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Initialized Threaded Earth workspace with population_size={loaded.simulation.population_size}")


@app.command()
def run(days: int = typer.Option(1, min=1), seed: int = typer.Option(42), config: Path = typer.Option(DEFAULT_CONFIG_PATH)) -> None:
    loaded = load_config(config)
    run_id = make_run_id(seed)
    ensure_artifact_dirs(run_id)
    init_db(db_path(run_id))
    SessionLocal = session_factory(db_path(run_id))
    with SessionLocal() as session:
        initialize_run(session, run_id, seed, loaded)
        run_simulation(session, run_id, days, seed, loaded)
        generate_report(session, run_id)
    typer.echo(f"run_id={run_id}")
    typer.echo(f"database={db_path(run_id)}")
    typer.echo(f"events={event_log_path(run_id)}")
    typer.echo(f"report={report_path(run_id)}")


@app.command()
def replay(run_id: str = typer.Option(..., help="Run id to replay.")) -> None:
    log_path = event_log_path(run_id)
    if not log_path.exists():
        raise typer.BadParameter(f"No event log found for {run_id}")
    for line in log_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        typer.echo(f"tick {record['tick']:>3} {record['event_type']}: {record['summary']}")


@app.command()
def report(run_id: str = typer.Option(..., help="Run id to report on.")) -> None:
    database = db_path(run_id)
    if not database.exists():
        raise typer.BadParameter(f"No database found for {run_id}")
    SessionLocal = session_factory(database)
    with SessionLocal() as session:
        path = generate_report(session, run_id)
    typer.echo(str(path))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run("threaded_earth.web:app", host=host, port=port, reload=False)


def record_manual_event(run_id: str, summary: str) -> None:
    """Tiny helper kept for future sandbox controls; not exposed in stage 1 CLI."""
    append_jsonl(event_log_path(run_id), {"run_id": run_id, "event_type": "manual_note", "summary": summary})
