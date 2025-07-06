# Iris ML Pipeline Makefile

.PHONY: help install install-dev train api web test lint format clean docker-build docker-run

# Variables
PYTHON := python
PIP := pip
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
MODELS_DIR := models
DATA_DIR := data

# Default target
help:
	@echo "Iris ML Pipeline - Available Commands:"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  train        - Train ML models"
	@echo "  enterprise   - Run enterprise ML pipeline with advanced features"
	@echo "  api          - Start FastAPI server"
	@echo "  web          - Start Streamlit web interface"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

# Installation
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev]

# Training
train:
	$(PYTHON) -m iris_pipeline.models.training

# Enterprise ML Pipeline
enterprise:
	$(PYTHON) scripts/run_enterprise_pipeline.py

# Services
api:
	uvicorn iris_pipeline.api.server:app --host 0.0.0.0 --port 8000 --reload

web:
	streamlit run apps/web_interface.py --server.port 8501

# Development
test:
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term

lint:
	flake8 $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR)

format:
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

# Maintenance
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Data
download-data:
	@echo "Data already available in $(DATA_DIR)/"

# Documentation
docs:
	cd $(DOCS_DIR) && make html

docs-serve:
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8080

# Docker
docker-build:
	docker build -t iris-ml-pipeline .

docker-run:
	docker run -p 8000:8000 -p 8501:8501 iris-ml-pipeline

# Production
deploy-api:
	gunicorn iris_pipeline.api.server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Health checks
health-check:
	curl -f http://localhost:8000/health || exit 1

# Setup development environment
setup-dev: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

# CI/CD helpers
ci-test: lint test
	@echo "CI tests passed!"

ci-build: clean install test
	@echo "CI build complete!"

# Database (if needed in future)
db-init:
	@echo "Database initialization not required for current version"

db-migrate:
	@echo "Database migration not required for current version"

# Monitoring
logs:
	tail -f logs/*.log 2>/dev/null || echo "No log files found"

# Quick start
quickstart: install train api

# Full demo
demo: train
	@echo "Starting demo environment..."
	@echo "1. Starting API server in background..."
	nohup make api > api.log 2>&1 &
	@sleep 5
	@echo "2. Starting web interface..."
	make web 