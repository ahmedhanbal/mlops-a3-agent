.PHONY: install lint test seed agent evidence
install:
	pip install -e ".[dev]"
lint:
	ruff check . && ruff format --check .
test:
	pytest -q
seed:            ## create DVC dataset v1/v2 + MLflow runs (demo state)
	./scripts/seed_demo.sh
agent:
	python -m agent.main
evidence:
	./scripts/collect_evidence.sh
