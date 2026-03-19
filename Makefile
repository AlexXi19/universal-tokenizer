PYTHON := .venv/bin/python
UV     := uv
PORT   ?= 8000
WORKERS?= 4

.PHONY: install test run dev prod format lint clean docker-build docker-run loadtest help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install dependencies
	$(UV) venv .venv
	$(UV) pip install -r requirements.txt
	$(UV) pip install ruff

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

dev: ## Run dev server (port=$(PORT))
	PORT=$(PORT) $(PYTHON) run.py

prod: ## Run production server with gunicorn
	WORKERS=$(WORKERS) .venv/bin/gunicorn --config gunicorn_config.py run:app

format: ## Format code with ruff
	.venv/bin/ruff format app/ tests/

lint: ## Lint and check formatting
	.venv/bin/ruff check app/ tests/
	.venv/bin/ruff format --check app/ tests/

loadtest: ## Run load test (requires locust)
	cd tests/loadtest && locust -f loadtest.py --host=http://localhost:8080 -u 50 -r 50 --run-time 1m

docker-build: ## Build Docker image
	docker build -t universal-tokenizer .

docker-run: ## Run Docker container
	docker run -p 8080:8080 -e WORKERS=$(WORKERS) universal-tokenizer

clean: ## Remove .venv and caches
	rm -rf .venv __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
