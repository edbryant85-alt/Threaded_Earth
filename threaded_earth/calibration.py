from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from threaded_earth.config import ThreadedEarthConfig
from threaded_earth.db import init_db, session_factory
from threaded_earth.diagnostics import diagnose_run
from threaded_earth.paths import (
    calibration_dir,
    calibration_json_path,
    calibration_report_path,
    db_path,
    ensure_artifact_dirs,
    run_dir,
)
from threaded_earth.reports import generate_report
from threaded_earth.simulation import initialize_run, run_simulation


def make_calibration_id(seeds: list[int], durations: list[int], profile: str = "default") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    seed_part = "-".join(str(seed) for seed in seeds)
    day_part = "-".join(str(day) for day in durations)
    return f"calibration-{stamp}-{profile}-days-{day_part}-seeds-{seed_part}"


def run_calibration(
    seeds: list[int],
    durations: list[int],
    config: ThreadedEarthConfig,
    profile: str = "default",
    calibration_id: str | None = None,
) -> dict[str, Any]:
    calibration_id = calibration_id or make_calibration_id(seeds, durations, profile)
    calibration_dir(calibration_id).mkdir(parents=True, exist_ok=True)

    runs = []
    diagnostics_by_run: dict[str, dict[str, Any]] = {}
    for days in durations:
        for seed in seeds:
            run_id = f"{calibration_id}-days-{days}-seed-{seed}"
            if run_dir(run_id).exists():
                shutil.rmtree(run_dir(run_id))
            ensure_artifact_dirs(run_id)
            init_db(db_path(run_id))
            SessionLocal = session_factory(db_path(run_id))
            with SessionLocal() as session:
                initialize_run(session, run_id, seed, config)
                run_simulation(session, run_id, days, seed, config)
                diagnostic = diagnose_run(session, run_id, config.simulation.diagnostics)
                generate_report(session, run_id)
            run_record = {
                "seed": seed,
                "days": days,
                "run_id": run_id,
                "warning_count": len(diagnostic["warnings"]),
                "severity_counts": _severity_counts(diagnostic["warnings"]),
            }
            runs.append(run_record)
            diagnostics_by_run[run_id] = diagnostic

    aggregate = _aggregate_calibration(calibration_id, profile, seeds, durations, runs, diagnostics_by_run)
    calibration_json_path(calibration_id).write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    calibration_report_path(calibration_id).write_text(_render_report(aggregate), encoding="utf-8")
    return aggregate


def _aggregate_calibration(
    calibration_id: str,
    profile: str,
    seeds: list[int],
    durations: list[int],
    runs: list[dict[str, Any]],
    diagnostics_by_run: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    all_warnings = [
        {**warning, "run_id": run_id}
        for run_id, diagnostic in diagnostics_by_run.items()
        for warning in diagnostic.get("warnings", [])
    ]
    warning_counts_by_type = Counter(warning["metric_name"] for warning in all_warnings)
    severity_counts = Counter(warning["severity"] for warning in all_warnings)
    recurring_findings = _recurring_findings(diagnostics_by_run, len(runs))
    recommendations = _parameter_recommendations(recurring_findings, warning_counts_by_type)
    status = "fail" if any(warning["severity"] == "critical" for warning in all_warnings) else "pass"
    if any(warning["severity"] == "warning" for warning in all_warnings):
        status = "warning"
    if any(finding["severity"] == "critical" for finding in recurring_findings):
        status = "fail"
    return {
        "calibration_id": calibration_id,
        "profile": profile,
        "seeds": seeds,
        "durations": durations,
        "run_ids": [run["run_id"] for run in runs],
        "runs": runs,
        "diagnostics_by_run": diagnostics_by_run,
        "warning_counts_by_type": dict(sorted(warning_counts_by_type.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "recurring_findings": recurring_findings,
        "parameter_recommendations": recommendations,
        "status": status,
    }


def _recurring_findings(diagnostics_by_run: dict[str, dict[str, Any]], run_count: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "severities": Counter(), "run_ids": []})
    for run_id, diagnostic in diagnostics_by_run.items():
        seen_in_run = {}
        for warning in diagnostic.get("warnings", []):
            metric_name = warning["metric_name"]
            current = seen_in_run.get(metric_name)
            if current is None or _severity_rank(warning["severity"]) > _severity_rank(current):
                seen_in_run[metric_name] = warning["severity"]
        for metric_name, severity in seen_in_run.items():
            grouped[metric_name]["count"] += 1
            grouped[metric_name]["severities"][severity] += 1
            grouped[metric_name]["run_ids"].append(run_id)

    threshold = max(2, (run_count + 1) // 2)
    findings = []
    for metric_name, data in grouped.items():
        if data["count"] < threshold:
            continue
        severity = _highest_severity(data["severities"])
        findings.append(
            {
                "metric_name": metric_name,
                "count": data["count"],
                "run_fraction": round(data["count"] / max(1, run_count), 3),
                "severity": severity,
                "run_ids": sorted(data["run_ids"]),
            }
        )
    return sorted(findings, key=lambda item: (-_severity_rank(item["severity"]), -item["count"], item["metric_name"]))


def _parameter_recommendations(recurring_findings: list[dict[str, Any]], counts: Counter[str]) -> list[dict[str, Any]]:
    mapped = {
        "total_food": (
            "food_balance",
            "If food collapse recurs, lower `daily_food_need_per_agent` or increase initial food/resource-seeking yield.",
        ),
        "total_materials": (
            "material_balance",
            "If material runaway recurs, enable `material_decay_enabled` or lower material gains.",
        ),
        "household_shortage": (
            "upkeep_pressure",
            "Persistent shortages point to daily upkeep outpacing food acquisition; tune food need, starting food, or resource-seeking yield.",
        ),
        "trust": ("relationship_deltas", "Relationship saturation suggests reducing trust deltas or adding mild decay."),
        "affinity": ("relationship_deltas", "Affinity saturation suggests reducing affinity deltas or adding mild decay."),
        "reputation": ("relationship_deltas", "Reputation saturation suggests reducing reputation deltas or adding mild decay."),
        "conflict": ("conflict_scoring", "Conflict runaway suggests lowering conflict base score or strengthening repair/cooperation modifiers."),
        "cooperation": ("cooperation_scoring", "Cooperation collapse suggests increasing cooperation scoring, valid targets, or resource-sharing incentives."),
        "social_propagation": ("propagation_pressure", "Propagation runaway suggests lowering propagation caps, observer count, or propagation strength."),
        "norm_support": ("norm_evidence", "Norm support saturation suggests lowering event contribution or increasing breadth requirements."),
        "role_signal": ("role_stabilization", "Role signal saturation suggests increasing role thresholds, adding role decay, or lowering role evidence increments."),
        "resource_exchange": ("dead_resource_exchange", "A dead exchange subsystem suggests checking sharing/trade action scores and resource preconditions."),
        "repair": ("dead_repair", "A dead repair subsystem suggests checking repair goal creation and target-aware repair scoring."),
    }
    recommendations = []
    recurring_names = {finding["metric_name"] for finding in recurring_findings}
    for metric_name in sorted(recurring_names | {name for name, count in counts.items() if count >= 2}):
        target, text = mapped.get(
            metric_name,
            (metric_name, f"Review `{metric_name}` thresholds and scoring because it recurred across calibration runs."),
        )
        recommendations.append(
            {
                "target": target,
                "metric_name": metric_name,
                "recommendation": text,
                "supporting_warning_count": counts.get(metric_name, 0),
            }
        )
    return recommendations


def _render_report(aggregate: dict[str, Any]) -> str:
    run_lines = [
        f"- days={run['days']} seed={run['seed']}: {run['run_id']} warnings={run['warning_count']} severities={run['severity_counts']}"
        for run in aggregate["runs"]
    ]
    summary_lines = [
        f"- {metric}: {count}"
        for metric, count in aggregate["warning_counts_by_type"].items()
    ]
    recurring_lines = [
        f"- {finding['severity']} {finding['metric_name']}: {finding['count']} runs ({finding['run_fraction']:.0%})"
        for finding in aggregate["recurring_findings"]
    ]
    recommendation_lines = [
        f"- {item['target']}: {item['recommendation']} ({item['supporting_warning_count']} warnings)"
        for item in aggregate["parameter_recommendations"]
    ]
    return "\n".join(
        [
            f"# Threaded Earth Calibration: {aggregate['calibration_id']}",
            "",
            "## Overview",
            f"- profile: {aggregate['profile']}",
            f"- status: {aggregate['status']}",
            f"- seeds: {aggregate['seeds']}",
            f"- durations: {aggregate['durations']}",
            "- scope: diagnostics-driven defaults check; descriptive only, no config mutation.",
            "",
            "## Run Matrix",
            *run_lines,
            "",
            "## Diagnostics Summary",
            *summary_lines,
            "",
            "## Recurring Failures",
            *(recurring_lines or ["- No recurring failures met the recurrence threshold."]),
            "",
            "## Parameter Recommendations",
            *(recommendation_lines or ["- No parameter recommendations generated."]),
            "",
            "## Suggested Next Test",
            "- Apply only conservative parameter changes, then rerun the same seed/duration matrix and compare diagnostics.",
            "",
            "## Limitations",
            "- This is threshold-based calibration, not automated optimization.",
            "- Recommendations are based on observed diagnostics and should be reviewed before changing defaults.",
            "- Norms and roles remain descriptive signals, not formal rules or institutions.",
            "",
        ]
    )


def _severity_counts(warnings: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(warning["severity"] for warning in warnings).items()))


def _highest_severity(counter: Counter[str]) -> str:
    return max(counter, key=_severity_rank)


def _severity_rank(severity: str) -> int:
    return {"info": 1, "warning": 2, "critical": 3}.get(severity, 0)
