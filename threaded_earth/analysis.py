from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.config import ThreadedEarthConfig
from threaded_earth.db import init_db, session_factory
from threaded_earth.metrics import compute_metrics
from threaded_earth.models import Event, NormCandidate, RoleSignal
from threaded_earth.paths import analysis_dir, analysis_json_path, analysis_report_path, db_path, ensure_artifact_dirs
from threaded_earth.propagation import propagation_stats
from threaded_earth.reports import generate_report
from threaded_earth.resources import household_resource_summary
from threaded_earth.roles import role_stats
from threaded_earth.simulation import initialize_run, run_simulation


def make_analysis_id(seeds: list[int], days: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"analysis-{stamp}-days-{days}-seeds-{'-'.join(str(seed) for seed in seeds)}"


def run_multi_seed_analysis(
    seeds: list[int],
    days: int,
    config: ThreadedEarthConfig,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    analysis_id = analysis_id or make_analysis_id(seeds, days)
    analysis_dir(analysis_id).mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        run_id = f"{analysis_id}-seed-{seed}"
        ensure_artifact_dirs(run_id)
        init_db(db_path(run_id))
        SessionLocal = session_factory(db_path(run_id))
        with SessionLocal() as session:
            initialize_run(session, run_id, seed, config)
            run_simulation(session, run_id, days, seed, config)
            generate_report(session, run_id)
            runs.append(_run_summary(session, run_id, seed))

    aggregate = _aggregate(analysis_id, seeds, days, runs)
    analysis_json_path(analysis_id).write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    analysis_report_path(analysis_id).write_text(_render_report(aggregate), encoding="utf-8")
    return aggregate


def _run_summary(session: Session, run_id: str, seed: int) -> dict[str, Any]:
    norms = session.query(NormCandidate).filter(NormCandidate.run_id == run_id).order_by(NormCandidate.norm_name).all()
    roles = role_stats(session, run_id)
    propagation = propagation_stats(session, run_id)
    resources = household_resource_summary(session, run_id)
    event_counts = _event_counts(session, run_id)
    return {
        "seed": seed,
        "run_id": run_id,
        "metrics": compute_metrics(session, run_id),
        "norm_candidates": [
            {
                "norm_name": norm.norm_name,
                "status": norm.status,
                "evidence_count": norm.evidence_count,
                "support_score": round(norm.support_score, 3),
                "opposition_score": round(norm.opposition_score, 3),
                "contributing_agent_count": len(norm.contributing_agent_ids or []),
                "contributing_household_count": len(norm.contributing_household_ids or []),
                "breadth_score": round(norm.breadth_score, 3),
            }
            for norm in norms
        ],
        "role_counts_above_threshold": roles["role_counts_above_threshold"],
        "top_role_signals": roles["top_role_signals"][:5],
        "propagation": propagation,
        "resources": {
            "total_food": resources["total_food"],
            "total_materials": resources["total_materials"],
            "households_below_scarcity_threshold": resources["households_below_scarcity_threshold"],
        },
        "event_counts": event_counts,
    }


def _aggregate(analysis_id: str, seeds: list[int], days: int, runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = sorted({key for run in runs for key in run["metrics"]})
    average_metrics = {key: _average([run["metrics"].get(key) for run in runs]) for key in metric_keys}
    metric_ranges = {
        key: _range([run["metrics"].get(key) for run in runs])
        for key in metric_keys
    }
    norm_occurrences: dict[str, dict[str, Any]] = {}
    for run in runs:
        for norm in run["norm_candidates"]:
            record = norm_occurrences.setdefault(
                norm["norm_name"],
                {"count": 0, "stable_count": 0, "support_scores": [], "breadth_scores": []},
            )
            record["count"] += 1
            record["stable_count"] += 1 if norm["status"] == "stable" else 0
            record["support_scores"].append(norm["support_score"])
            record["breadth_scores"].append(norm["breadth_score"])
    recurring_norms = [
        {
            "norm_name": name,
            "run_count": data["count"],
            "stable_count": data["stable_count"],
            "average_support": _average(data["support_scores"]),
            "average_breadth": _average(data["breadth_scores"]),
        }
        for name, data in sorted(norm_occurrences.items())
        if data["count"] >= 2
    ]
    return {
        "analysis_id": analysis_id,
        "days": days,
        "seeds": seeds,
        "run_ids": [run["run_id"] for run in runs],
        "runs": runs,
        "average_metrics": average_metrics,
        "metric_ranges": metric_ranges,
        "recurring_norms": recurring_norms,
        "role_signal_summaries": _role_summary(runs),
        "propagation_pressure_summaries": _propagation_summary(runs),
        "resource_scarcity_summaries": _resource_summary(runs),
        "conflict_cooperation_summaries": _conflict_cooperation_summary(runs),
        "notable_variance": _notable_variance(metric_ranges),
    }


def _render_report(aggregate: dict[str, Any]) -> str:
    run_lines = [f"- seed {run['seed']}: {run['run_id']}" for run in aggregate["runs"]]
    metric_lines = [f"- {key}: avg={value}" for key, value in aggregate["average_metrics"].items()]
    norm_lines = [
        f"- {norm['norm_name']}: runs={norm['run_count']}; stable={norm['stable_count']}; "
        f"avg_support={norm['average_support']}; avg_breadth={norm['average_breadth']}"
        for norm in aggregate["recurring_norms"]
    ]
    variance_lines = [
        f"- {item['metric']}: min={item['min']} max={item['max']} spread={item['spread']}"
        for item in aggregate["notable_variance"]
    ]
    return "\n".join(
        [
            f"# Threaded Earth Multi-Seed Analysis: {aggregate['analysis_id']}",
            "",
            "## Seeds And Runs",
            *run_lines,
            "",
            "## Average Metrics",
            *metric_lines,
            "",
            "## Recurring Norm Candidates",
            *(norm_lines or ["- No recurring norms detected across at least two seeds."]),
            "",
            "## Role Signal Summaries",
            *[f"- {key}: {value}" for key, value in aggregate["role_signal_summaries"].items()],
            "",
            "## Propagation Pressure Summaries",
            *[f"- {key}: {value}" for key, value in aggregate["propagation_pressure_summaries"].items()],
            "",
            "## Resource Scarcity Summaries",
            *[f"- {key}: {value}" for key, value in aggregate["resource_scarcity_summaries"].items()],
            "",
            "## Conflict Cooperation Summaries",
            *[f"- {key}: {value}" for key, value in aggregate["conflict_cooperation_summaries"].items()],
            "",
            "## Notable Variance",
            *(variance_lines or ["- No metric variance detected."]),
            "",
        ]
    )


def _event_counts(session: Session, run_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event_type, in session.query(Event.event_type).filter(Event.run_id == run_id).all():
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _role_summary(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        for role_name, count in run["role_counts_above_threshold"].items():
            totals[role_name] = totals.get(role_name, 0) + count
    return dict(sorted(totals.items()))


def _propagation_summary(runs: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "average_propagated_events": _average([run["propagation"]["propagated_event_count"] for run in runs]),
        "average_propagated_memories": _average([run["propagation"]["propagated_memory_count"] for run in runs]),
        "average_skipped_propagation": _average([run["propagation"]["propagation_skipped_count"] for run in runs]),
    }


def _resource_summary(runs: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "average_total_food": _average([run["resources"]["total_food"] for run in runs]),
        "average_scarce_households": _average([run["resources"]["households_below_scarcity_threshold"] for run in runs]),
    }


def _conflict_cooperation_summary(runs: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "average_conflicts": _average([run["event_counts"].get("conflict", 0) for run in runs]),
        "average_cooperations": _average([run["event_counts"].get("cooperation", 0) for run in runs]),
        "average_resource_exchanges": _average([run["event_counts"].get("resource_exchange", 0) for run in runs]),
    }


def _notable_variance(ranges: dict[str, dict[str, float]]) -> list[dict[str, float]]:
    rows = [
        {"metric": key, **value}
        for key, value in ranges.items()
        if value["spread"] > 0
    ]
    return sorted(rows, key=lambda item: (-item["spread"], item["metric"]))[:8]


def _average(values: list[Any]) -> float:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return 0.0
    return round(sum(numeric) / len(numeric), 5)


def _range(values: list[Any]) -> dict[str, float]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"min": 0.0, "max": 0.0, "spread": 0.0}
    return {"min": round(min(numeric), 5), "max": round(max(numeric), 5), "spread": round(max(numeric) - min(numeric), 5)}
