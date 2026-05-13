from __future__ import annotations

import json

from typer.testing import CliRunner

from threaded_earth.analysis import run_multi_seed_analysis
from threaded_earth.cli import app
from threaded_earth.config import load_config
from threaded_earth.paths import analysis_json_path, analysis_report_path, db_path


def test_analyze_command_creates_runs_and_analysis_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.cli.ARTIFACTS_DIR", tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["analyze", "--seeds", "1,2", "--days", "3"])

    assert result.exit_code == 0, result.output
    analysis_id = [line for line in result.output.splitlines() if line.startswith("analysis_id=")][0].split("=", 1)[1]
    data = json.loads(analysis_json_path(analysis_id).read_text(encoding="utf-8"))
    assert analysis_report_path(analysis_id).exists()
    assert data["seeds"] == [1, 2]
    assert len(data["run_ids"]) == 2
    assert all(db_path(run_id).exists() for run_id in data["run_ids"])
    assert "average_metrics" in data
    assert "recurring_norms" in data


def test_multi_seed_analysis_detects_recurring_norms_and_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()

    first = run_multi_seed_analysis([3, 4], 3, config, analysis_id="analysis-fixed")

    norm_names = [norm["norm_name"] for norm in first["recurring_norms"]]
    assert first["run_ids"] == ["analysis-fixed-seed-3", "analysis-fixed-seed-4"]
    assert norm_names
    assert first["average_metrics"]
    assert first["propagation_pressure_summaries"]
    assert first["resource_scarcity_summaries"]
    assert first["conflict_cooperation_summaries"]
