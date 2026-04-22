.PHONY: test lint format docker-build docker-up docker-down clean

test:
	python -m pytest tests/ -x --timeout=60

lint:
	ruff check .

format:
	ruff format .

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf logs/*.log 2>/dev/null || true
