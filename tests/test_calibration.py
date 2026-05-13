from __future__ import annotations

import json

from typer.testing import CliRunner

from threaded_earth.calibration import _aggregate_calibration, run_calibration
from threaded_earth.cli import app
from threaded_earth.config import load_config
from threaded_earth.paths import calibration_json_path, calibration_report_path, db_path


def test_calibrate_command_creates_expected_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("threaded_earth.cli.ARTIFACTS_DIR", tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["calibrate", "--seeds", "1,2", "--days", "1,2", "--output-id", "calibration-test"],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(calibration_json_path("calibration-test").read_text(encoding="utf-8"))
    assert calibration_report_path("calibration-test").exists()
    assert data["calibration_id"] == "calibration-test"
    assert data["seeds"] == [1, 2]
    assert data["durations"] == [1, 2]
    assert len(data["run_ids"]) == 4
    assert all(db_path(run_id).exists() for run_id in data["run_ids"])
    assert "diagnostics_by_run" in data
    assert "warning_counts_by_type" in data
    assert "parameter_recommendations" in data


def test_calibration_aggregates_recurring_findings_and_recommendations():
    diagnostics = {
        "run-a": {
            "run_id": "run-a",
            "warnings": [
                _warning("critical", "total_food"),
                _warning("warning", "household_shortage"),
            ],
        },
        "run-b": {
            "run_id": "run-b",
            "warnings": [
                _warning("critical", "total_food"),
                _warning("warning", "total_materials"),
            ],
        },
        "run-c": {
            "run_id": "run-c",
            "warnings": [
                _warning("warning", "household_shortage"),
                _warning("warning", "total_materials"),
                _warning("info", "role_signal"),
            ],
        },
    }
    runs = [
        {"seed": 1, "days": 15, "run_id": "run-a", "warning_count": 2, "severity_counts": {"critical": 1, "warning": 1}},
        {"seed": 2, "days": 15, "run_id": "run-b", "warning_count": 2, "severity_counts": {"critical": 1, "warning": 1}},
        {"seed": 3, "days": 15, "run_id": "run-c", "warning_count": 2, "severity_counts": {"info": 1, "warning": 1}},
    ]

    aggregate = _aggregate_calibration("calibration-known", "default", [1, 2, 3], [15], runs, diagnostics)

    recurring = {finding["metric_name"]: finding for finding in aggregate["recurring_findings"]}
    recommendations = {item["metric_name"]: item["recommendation"] for item in aggregate["parameter_recommendations"]}
    assert recurring["total_food"]["severity"] == "critical"
    assert recurring["household_shortage"]["count"] == 2
    assert "daily_food_need_per_agent" in recommendations["total_food"]
    assert "material_decay_enabled" in recommendations["total_materials"]
    assert aggregate["status"] == "fail"


def test_calibration_is_deterministic_for_fixed_id(tmp_path, monkeypatch):
    monkeypatch.setattr("threaded_earth.paths.ARTIFACTS_DIR", tmp_path)
    config = load_config()

    first = run_calibration([3], [1], config, calibration_id="calibration-fixed")
    second = run_calibration([3], [1], config, calibration_id="calibration-fixed")

    assert first == second


def _warning(severity: str, metric_name: str) -> dict:
    return {
        "severity": severity,
        "metric_name": metric_name,
        "evidence": {},
        "evidence_text": f"{metric_name} evidence",
        "recommendation": f"{metric_name} recommendation",
        "affected_ticks": [],
    }
