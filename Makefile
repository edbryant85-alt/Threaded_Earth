.PHONY: install test init run report serve iteration checkpoint checkpoint-no-push checkpoint-dry-run clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

init:
	threaded-earth init

run:
	threaded-earth run --days 5 --seed 42

report:
	threaded-earth report --run-id "$$RUN_ID"

serve:
	threaded-earth serve

iteration: test checkpoint

checkpoint:
	python tools/checkpoint.py

checkpoint-no-push:
	python tools/checkpoint.py --no-push

checkpoint-dry-run:
	python tools/checkpoint.py --dry-run

clean:
	rm -rf artifacts .pytest_cache
