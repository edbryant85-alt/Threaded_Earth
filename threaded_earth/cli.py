from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from threaded_earth.analysis import run_multi_seed_analysis
from threaded_earth.config import load_config
from threaded_earth.db import init_db, session_factory
from threaded_earth.diagnostics import diagnose_analysis, diagnose_run
from threaded_earth.events import append_jsonl
from threaded_earth.models import Event
from threaded_earth.paths import (
    ARTIFACTS_DIR,
    DEFAULT_CONFIG_PATH,
    analysis_diagnostics_json_path,
    analysis_diagnostics_report_path,
    db_path,
    diagnostics_json_path,
    diagnostics_report_path,
    ensure_artifact_dirs,
    event_log_path,
    report_path,
)
from threaded_earth.reports import generate_report
from threaded_earth.snapshots import snapshot_inventory
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
    database = db_path(run_id)
    if database.exists():
        SessionLocal = session_factory(database)
        with SessionLocal() as session:
            inventory = snapshot_inventory(session, run_id)
        expected = inventory.expected_ticks if inventory.expected_ticks is not None else "unknown"
        latest = inventory.latest_tick if inventory.latest_tick is not None else "none"
        typer.echo(
            f"snapshot_replay={inventory.status} snapshots={inventory.count} "
            f"expected_ticks={expected} latest_tick={latest}"
        )
    else:
        typer.echo("snapshot_replay=unavailable snapshots=0 expected_ticks=unknown latest_tick=none")
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
def analyze(
    seeds: str = typer.Option(..., help="Comma-separated integer seeds, e.g. 1,2,3."),
    days: int = typer.Option(15, min=1),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH),
) -> None:
    seed_values = _parse_seeds(seeds)
    loaded = load_config(config)
    aggregate = run_multi_seed_analysis(seed_values, days, loaded)
    typer.echo(f"analysis_id={aggregate['analysis_id']}")
    typer.echo(f"analysis_json={ARTIFACTS_DIR / 'analysis' / aggregate['analysis_id'] / 'analysis.json'}")
    typer.echo(f"analysis_report={ARTIFACTS_DIR / 'analysis' / aggregate['analysis_id'] / 'analysis.md'}")
    for run_id in aggregate["run_ids"]:
        typer.echo(f"run_id={run_id}")


@app.command()
def diagnose(
    run_id: str = typer.Option(..., help="Run id to diagnose."),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH),
) -> None:
    database = db_path(run_id)
    if not database.exists():
        raise typer.BadParameter(f"No database found for {run_id}")
    loaded = load_config(config)
    SessionLocal = session_factory(database)
    with SessionLocal() as session:
        result = diagnose_run(session, run_id, loaded.simulation.diagnostics)
    typer.echo(f"diagnostics_warnings={len(result['warnings'])}")
    typer.echo(f"diagnostics_json={diagnostics_json_path(run_id)}")
    typer.echo(f"diagnostics_report={diagnostics_report_path(run_id)}")


@app.command("diagnose-analysis")
def diagnose_analysis_command(
    analysis_id: str = typer.Option(..., help="Analysis id to diagnose."),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH),
) -> None:
    loaded = load_config(config)
    result = diagnose_analysis(analysis_id, loaded.simulation.diagnostics)
    typer.echo(f"diagnostics_warnings={len(result['warnings'])}")
    typer.echo(f"diagnostics_json={analysis_diagnostics_json_path(analysis_id)}")
    typer.echo(f"diagnostics_report={analysis_diagnostics_report_path(analysis_id)}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run("threaded_earth.web:app", host=host, port=port, reload=False)


def record_manual_event(run_id: str, summary: str) -> None:
    """Tiny helper kept for future sandbox controls; not exposed in stage 1 CLI."""
    append_jsonl(event_log_path(run_id), {"run_id": run_id, "event_type": "manual_note", "summary": summary})


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if not seeds:
        raise typer.BadParameter("At least one seed is required.")
    return seeds
