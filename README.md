# Threaded Earth

Threaded Earth is a local-first synthetic society simulation foundation. This first vertical slice models one Neolithic-inspired river valley settlement with symbolic, inspectable agents. It is not a game and it does not call LLMs or external services.

The current goal is modest: create replayable state, logs, decisions, reports, and a dashboard before adding richer society systems.

## Setup

```bash
python -m pip install -e ".[dev]"
```

In GitHub Codespaces, the devcontainer runs that install command automatically.

## Commands

```bash
threaded-earth init
threaded-earth run --days 5 --seed 42
threaded-earth replay --run-id RUN_ID
threaded-earth report --run-id RUN_ID
threaded-earth serve
pytest
make iteration
make checkpoint
```

Sample runs write artifacts to:

```text
artifacts/<run_id>/
  threaded_earth.sqlite
  logs/events.jsonl
  reports/report.md
  exports/metrics.json
  snapshots/tick_<N>.json
```

## Architecture Notes

- `threaded_earth/models.py`: SQLite persistence schema using SQLAlchemy.
- `threaded_earth/generation.py`: seeded household, kinship, relationship, resource, and agent creation.
- `threaded_earth/cognition.py`: transparent symbolic decision rules with candidate actions and confidence.
- `threaded_earth/targeting.py`: explicit target selection for social actions.
- `threaded_earth/resources.py`: simple household-level resource accounting.
- `threaded_earth/simulation.py`: one daily tick loop with explicit state transitions.
- `threaded_earth/events.py`: DB event creation plus JSONL logging.
- `threaded_earth/reports.py`: Markdown research report generation.
- `threaded_earth/snapshots.py`: compact per-tick state snapshots and metric deltas.
- `threaded_earth/web.py`: simple FastAPI HTML dashboard.
- `threaded_earth/cli.py`: Typer command surface.
- `tools/checkpoint.py`: stages source changes, generates a local commit message, commits, and pushes.

## Iteration Checkpoints

After each coding iteration, run:

```bash
make iteration
```

This runs the test suite first. If tests pass, it stages non-ignored changes, generates a commit message from the changed files, commits, and pushes to the current branch on `origin`.

To commit and push without running tests:

```bash
make checkpoint
```

Safer variants:

```bash
make checkpoint-dry-run
make checkpoint-no-push
```

Generated simulation artifacts under `artifacts/` are ignored by default. That is deliberate: run databases and logs are research outputs, while source commits should stay reviewable.

## Current Scope

Implemented:

- one symbolic river valley settlement
- approximately 50 agents
- households and kinship-style relationships
- simple needs, cooperation, conflict, resource exchange, food seeking, and rest
- inspectable decision logs
- basic emergence metrics
- local SQLite and JSONL artifacts

Not implemented:

- LLM cognition
- governance, religion, warfare, institutions, advanced ecology, advanced economics
- multiple settlements
- real-time simulation
- distributed systems, vector databases, queues, cloud services

## Ethics

Simulated agents are not assumed conscious. The simulator still treats welfare-relevant dynamics seriously: conflicts and resource pressure are explicit, logged, and reportable rather than hidden in narrative summaries.

## Troubleshooting

If the CLI is unavailable, reinstall the package:

```bash
python -m pip install -e ".[dev]"
```

If the dashboard shows no runs, create one first:

```bash
threaded-earth run --days 1 --seed 42
threaded-earth serve
```

If `make iteration` fails during push with a Git LFS message, rebuild the Codespace so the devcontainer Git LFS feature installs. This repository has an LFS pre-push hook, so push automation needs `git-lfs` available.
