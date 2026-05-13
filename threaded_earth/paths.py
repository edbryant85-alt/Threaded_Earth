from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def run_dir(run_id: str) -> Path:
    return ARTIFACTS_DIR / run_id


def db_path(run_id: str) -> Path:
    return run_dir(run_id) / "threaded_earth.sqlite"


def event_log_path(run_id: str) -> Path:
    return run_dir(run_id) / "logs" / "events.jsonl"


def report_path(run_id: str) -> Path:
    return run_dir(run_id) / "reports" / "report.md"


def metrics_path(run_id: str) -> Path:
    return run_dir(run_id) / "exports" / "metrics.json"


def snapshots_dir(run_id: str) -> Path:
    return run_dir(run_id) / "snapshots"


def snapshot_path(run_id: str, tick: int) -> Path:
    return snapshots_dir(run_id) / f"tick_{tick}.json"


def analysis_dir(analysis_id: str) -> Path:
    return ARTIFACTS_DIR / "analysis" / analysis_id


def analysis_report_path(analysis_id: str) -> Path:
    return analysis_dir(analysis_id) / "analysis.md"


def analysis_json_path(analysis_id: str) -> Path:
    return analysis_dir(analysis_id) / "analysis.json"


def ensure_artifact_dirs(run_id: str) -> None:
    root = run_dir(run_id)
    for child in ("logs", "reports", "exports", "snapshots"):
        (root / child).mkdir(parents=True, exist_ok=True)
