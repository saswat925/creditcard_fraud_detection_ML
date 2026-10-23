.PHONY: install lint format test train train-quick serve docker-build docker-run

install:
	pip install -e ".[notebook,dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

# Full run on the real dataset (data/creditcard.csv). Takes several minutes.
train:
	python -m fraud_detection.train

# Fast smoke run on synthetic data. Metrics are NOT meaningful.
train-quick:
	python -m fraud_detection.train --synthetic --quick

serve:
	uvicorn fraud_detection.api:app --reload

docker-build:
	docker build -t fraud-detection-api .

docker-run:
	docker run --rm -p 8000:8000 -v "$(CURDIR)/models:/app/models" fraud-detection-api
