.PHONY: install dev lint format test jobs models

install:
	pip install -e ".[train]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

models:   ## YOLO weights + plot fonts (run on a login node)
	bash scripts/download_models.sh

jobs:     ## render all Slurm scripts into slurm/generated/
	python3 slurm/generate.py --all
