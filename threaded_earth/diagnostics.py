from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.config import DiagnosticsConfig
from threaded_earth.models import Event, NormCandidate, Relationship, RoleSignal
from threaded_earth.paths import (
    analysis_diagnostics_json_path,
    analysis_diagnostics_report_path,
    analysis_json_path,
    diagnostics_json_path,
    diagnostics_report_path,
)
from threaded_earth.snapshots import load_snapshots


def diagnose_run(session: Session, run_id: str, config: DiagnosticsConfig | None = None) -> dict[str, Any]:
    config = config or DiagnosticsConfig()
    warnings: list[dict[str, Any]] = []
    if not config.diagnostics_enabled:
        result = {"run_id": run_id, "warnings": []}
        _write_run_outputs(run_id, result)
        return result

    snapshots = load_snapshots(run_id)
    events = session.query(Event).filter(Event.run_id == run_id).all()
    relationships = session.query(Relationship).filter(Relationship.run_id == run_id).all()
    norms = session.query(NormCandidate).filter(NormCandidate.run_id == run_id).all()
    roles = session.query(RoleSignal).filter(RoleSignal.run_id == run_id).all()

    warnings.extend(_resource_warnings(snapshots, config))
    warnings.extend(_shortage_warnings(snapshots, config))
    warnings.extend(_relationship_saturation_warnings(relationships, config))
    warnings.extend(_event_rate_warnings(snapshots, events, config))
    warnings.extend(_propagation_warnings(snapshots, config))
    warnings.extend(_norm_warnings(norms, config))
    warnings.extend(_role_warnings(roles, config))
    warnings.extend(_dead_system_warnings(events, norms, roles, config))
    warnings.extend(_flatline_warnings(snapshots, config))

    if not warnings:
        warnings.append(
            _warning(
                "info",
                "diagnostics",
                "No threshold warnings detected.",
                "Continue longer-run validation across multiple seeds.",
            )
        )
    result = {"run_id": run_id, "warnings": warnings}
    _write_run_outputs(run_id, result)
    return result


def diagnose_analysis(analysis_id: str, config: DiagnosticsConfig | None = None) -> dict[str, Any]:
    config = config or DiagnosticsConfig()
    data = json.loads(analysis_json_path(analysis_id).read_text(encoding="utf-8"))
    warnings: list[dict[str, Any]] = []
    recurring: dict[str, int] = {}
    seed_failures = []
    for run in data.get("runs", []):
        for norm in run.get("norm_candidates", []):
            if norm.get("status") == "stable":
                recurring[norm["norm_name"]] = recurring.get(norm["norm_name"], 0) + 1
        scarce = run.get("resources", {}).get("households_below_scarcity_threshold", 0)
        if scarce:
            seed_failures.append({"seed": run["seed"], "run_id": run["run_id"], "scarce_households": scarce})
    for metric, values in data.get("metric_ranges", {}).items():
        if values.get("spread", 0) > 0:
            warnings.append(
                _warning(
                    "info",
                    metric,
                    f"Metric varied across seeds with spread {values['spread']}.",
                    "Use the high-variance metrics as calibration targets.",
                    evidence=values,
                )
            )
    for item in seed_failures:
        warnings.append(
            _warning(
                "warning",
                "resource_scarcity",
                f"Seed {item['seed']} ended with {item['scarce_households']} scarce households.",
                "Compare food production, sharing, and upkeep pressure for this seed.",
                evidence=item,
            )
        )
    result = {
        "analysis_id": analysis_id,
        "warnings": warnings
        or [_warning("info", "analysis", "No aggregate diagnostics warnings detected.", "Run longer horizons.")],
        "recurring_warnings": {
            name: count for name, count in sorted(recurring.items()) if count >= 2
        },
        "seed_specific_failures": seed_failures,
    }
    analysis_diagnostics_json_path(analysis_id).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    analysis_diagnostics_report_path(analysis_id).write_text(_render_report(result), encoding="utf-8")
    return result


def _resource_warnings(snapshots: list[dict[str, Any]], config: DiagnosticsConfig) -> list[dict[str, Any]]:
    if len(snapshots) < 2:
        return []
    first = snapshots[0].get("household_resource_summary", {})
    last = snapshots[-1].get("household_resource_summary", {})
    warnings = []
    for key in ["total_food", "total_materials"]:
        start = float(first.get(key, 0) or 0)
        end = float(last.get(key, 0) or 0)
        if start > 0 and end / start >= config.runaway_growth_ratio_threshold:
            warnings.append(_warning("warning", key, f"{key} grew from {start:.2f} to {end:.2f}.", "Lower production or increase upkeep.", evidence={"start": start, "end": end}))
        if start > 0 and end / start <= config.collapse_ratio_threshold:
            warnings.append(_warning("critical", key, f"{key} collapsed from {start:.2f} to {end:.2f}.", "Increase production or reduce consumption pressure.", evidence={"start": start, "end": end}))
    return warnings


def _shortage_warnings(snapshots: list[dict[str, Any]], config: DiagnosticsConfig) -> list[dict[str, Any]]:
    if not snapshots:
        return []
    shortage_ticks = [
        snapshot["tick"]
        for snapshot in snapshots
        if (snapshot.get("upkeep_summary") or {}).get("households_with_shortage", 0) > 0
    ]
    rate = len(shortage_ticks) / len(snapshots)
    if rate >= config.shortage_rate_warning_threshold:
        return [_warning("warning", "household_shortage", f"Shortages occurred on {len(shortage_ticks)} of {len(snapshots)} ticks.", "Tune food production, sharing, or daily upkeep.", affected_ticks=shortage_ticks, evidence={"rate": round(rate, 3)})]
    return []


def _relationship_saturation_warnings(relationships: list[Relationship], config: DiagnosticsConfig) -> list[dict[str, Any]]:
    warnings = []
    for field in ["trust", "affinity", "reputation"]:
        values = [float(getattr(rel, field)) for rel in relationships]
        if not values:
            continue
        high = sum(1 for value in values if value >= config.saturation_high_threshold) / len(values)
        low = sum(1 for value in values if value <= config.saturation_low_threshold) / len(values)
        if high >= 0.4:
            warnings.append(_warning("warning", field, f"{field} has {high:.0%} values near maximum.", "Reduce positive deltas or add decay."))
        if low >= 0.4:
            warnings.append(_warning("warning", field, f"{field} has {low:.0%} values near minimum.", "Reduce negative deltas or add repair pressure."))
    return warnings


def _event_rate_warnings(snapshots: list[dict[str, Any]], events: list[Event], config: DiagnosticsConfig) -> list[dict[str, Any]]:
    days = max(1, max((snapshot.get("tick", 0) for snapshot in snapshots), default=1))
    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    warnings = []
    conflict_rate = counts.get("conflict", 0) / days
    if conflict_rate >= config.conflict_rate_warning_threshold:
        warnings.append(_warning("warning", "conflict", f"Conflict rate is {conflict_rate:.2f} per tick.", "Lower conflict scoring or increase repair/cooperation stabilizers."))
    if counts.get("cooperation", 0) <= config.min_activity_threshold:
        warnings.append(_warning("critical", "cooperation", "Cooperation activity collapsed or never started.", "Check cooperation scoring, targets, and needs."))
    return warnings


def _propagation_warnings(snapshots: list[dict[str, Any]], config: DiagnosticsConfig) -> list[dict[str, Any]]:
    if not snapshots:
        return []
    values = [(snapshot.get("propagation_summary") or {}).get("propagation_events_this_tick", 0) for snapshot in snapshots]
    average = sum(values) / len(values)
    if average >= config.propagation_rate_warning_threshold:
        return [_warning("warning", "social_propagation", f"Average propagation volume is {average:.2f} per tick.", "Lower propagation caps or observer counts.", evidence={"average": round(average, 3)})]
    return []


def _norm_warnings(norms: list[NormCandidate], config: DiagnosticsConfig) -> list[dict[str, Any]]:
    saturated = [norm.norm_name for norm in norms if norm.support_score >= 20]
    if saturated:
        return [_warning("info", "norm_support", f"Norm support saturation for: {', '.join(saturated[:5])}.", "Consider lowering event contribution or increasing breadth requirements.")]
    return []


def _role_warnings(roles: list[RoleSignal], config: DiagnosticsConfig) -> list[dict[str, Any]]:
    saturated = [role.role_signal_id for role in roles if role.score >= config.saturation_high_threshold]
    if saturated:
        return [_warning("info", "role_signal", f"{len(saturated)} role signals are near maximum.", "Consider role decay or lower evidence increments.")]
    return []


def _dead_system_warnings(events: list[Event], norms: list[NormCandidate], roles: list[RoleSignal], config: DiagnosticsConfig) -> list[dict[str, Any]]:
    counts = {event.event_type for event in events}
    warnings = []
    for event_type in ["resource_exchange", "repair", "conflict", "social_propagation"]:
        if event_type not in counts:
            warnings.append(_warning("warning", event_type, f"No {event_type} events recorded.", "Check whether this subsystem is unreachable."))
    if len(norms) <= config.min_activity_threshold:
        warnings.append(_warning("warning", "norm_candidates", "Norm candidate subsystem has little activity.", "Check norm evidence extraction."))
    if len(roles) <= config.min_activity_threshold:
        warnings.append(_warning("warning", "role_signals", "Role signal subsystem has little activity.", "Check role evidence updates."))
    return warnings


def _flatline_warnings(snapshots: list[dict[str, Any]], config: DiagnosticsConfig) -> list[dict[str, Any]]:
    if len(snapshots) < config.flatline_window_ticks:
        return []
    warnings = []
    for metric in ["resource_stress", "relationship_density", "reputation_variance"]:
        values = [snapshot.get("metrics", {}).get(metric) for snapshot in snapshots[-config.flatline_window_ticks :]]
        if len(set(values)) == 1:
            warnings.append(_warning("info", metric, f"{metric} flatlined for {config.flatline_window_ticks} ticks.", "Inspect whether this variable is expected to be static.", affected_ticks=[snapshot["tick"] for snapshot in snapshots[-config.flatline_window_ticks :]]))
    return warnings


def _write_run_outputs(run_id: str, result: dict[str, Any]) -> None:
    diagnostics_json_path(run_id).parent.mkdir(parents=True, exist_ok=True)
    diagnostics_report_path(run_id).parent.mkdir(parents=True, exist_ok=True)
    diagnostics_json_path(run_id).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    diagnostics_report_path(run_id).write_text(_render_report(result), encoding="utf-8")


def _render_report(result: dict[str, Any]) -> str:
    title_id = result.get("run_id") or result.get("analysis_id")
    by_severity: dict[str, int] = {}
    for warning in result["warnings"]:
        by_severity[warning["severity"]] = by_severity.get(warning["severity"], 0) + 1
    lines = [
        f"# Stability Diagnostics: {title_id}",
        "",
        "## Warnings By Severity",
        *[f"- {severity}: {count}" for severity, count in sorted(by_severity.items())],
        "",
        "## Findings",
    ]
    for warning in result["warnings"]:
        lines.append(
            f"- {warning['severity']} {warning['metric_name']}: {warning['evidence_text']} Recommendation: {warning['recommendation']}"
        )
    lines.extend(["", "## Next Calibration Target", _next_target(result["warnings"]), ""])
    return "\n".join(lines)


def _next_target(warnings: list[dict[str, Any]]) -> str:
    for severity in ["critical", "warning", "info"]:
        for warning in warnings:
            if warning["severity"] == severity:
                return f"- Start with `{warning['metric_name']}`: {warning['recommendation']}"
    return "- No calibration target selected."


def _warning(
    severity: str,
    metric_name: str,
    evidence_text: str,
    recommendation: str,
    evidence: dict[str, Any] | None = None,
    affected_ticks: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "metric_name": metric_name,
        "evidence": evidence or {},
        "evidence_text": evidence_text,
        "recommendation": recommendation,
        "affected_ticks": affected_ticks or [],
    }
