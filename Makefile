# Madness Matrix — unified developer interface
# Usage: make <target>
#
# Quick start:
#   make setup        # first-time install (backend + frontend)
#   make dev          # start backend API + Next.js dev server
#   make pipeline     # run full 6AM pipeline locally
#   make test         # run all tests

.PHONY: help setup dev dev-api dev-frontend \
        pipeline pipeline-intel pipeline-results \
        test test-backend test-frontend test-coverage \
        validate cache sweep \
        build clean

# ── Defaults ─────────────────────────────────────────────────────────────────
PYTHON     ?= python
SEASON     ?= 2026
PORT_API   ?= 8000
PORT_WEB   ?= 3000

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@printf "\n\033[1mMadness Matrix\033[0m — NCAA bracket prediction engine\n\n"
	@printf "\033[4mSetup\033[0m\n"
	@printf "  make setup            First-time install (Python + Node deps)\n"
	@printf "  make validate         Check all data source connections\n"
	@printf "  make cache            Pre-populate T-Rank cache for current season\n\n"
	@printf "\033[4mDevelopment\033[0m\n"
	@printf "  make dev              Start API server + frontend (parallel)\n"
	@printf "  make dev-api          Start FastAPI server only  (port $(PORT_API))\n"
	@printf "  make dev-frontend     Start Next.js dev server   (port $(PORT_WEB))\n\n"
	@printf "\033[4mPipeline\033[0m\n"
	@printf "  make pipeline         Run full 6 AM pipeline (scrape+train+write)\n"
	@printf "  make pipeline-intel   Run 12 PM intel update\n"
	@printf "  make pipeline-results Run 10 PM results ingestion\n\n"
	@printf "\033[4mTesting\033[0m\n"
	@printf "  make test             Run backend + frontend tests\n"
	@printf "  make test-backend     pytest only\n"
	@printf "  make test-frontend    vitest only\n"
	@printf "  make test-coverage    pytest with coverage report\n\n"
	@printf "\033[4mML\033[0m\n"
	@printf "  make train            Train ST-GNN locally (ADVI, season=$(SEASON))\n"
	@printf "  make sweep            Launch W&B hyperparameter sweep\n\n"
	@printf "\033[4mOther\033[0m\n"
	@printf "  make build            Build Next.js for production\n"
	@printf "  make clean            Remove caches and build artifacts\n\n"

# ── First-time setup ─────────────────────────────────────────────────────────
setup:
	@echo "→ Installing Python dependencies..."
	$(PYTHON) -m pip install -e ".[dev]" --quiet
	@echo "→ Installing Node dependencies..."
	cd frontend && npm install --silent
	@echo "→ Checking for .env file..."
	@test -f .env || (cp .env.example .env && echo "  ⚠️  .env created from .env.example — fill in your API keys")
	@echo "✅ Setup complete. Run 'make dev' to start."

# ── Development servers ───────────────────────────────────────────────────────
dev:
	@echo "→ Starting API (port $(PORT_API)) + frontend (port $(PORT_WEB))..."
	@trap 'kill 0' INT; \
	  $(PYTHON) -m uvicorn src.api.server:app --port $(PORT_API) --reload & \
	  cd frontend && npm run dev -- --port $(PORT_WEB) & \
	  wait

dev-api:
	$(PYTHON) -m uvicorn src.api.server:app --port $(PORT_API) --reload

dev-frontend:
	cd frontend && npm run dev -- --port $(PORT_WEB)

# ── Pipeline ─────────────────────────────────────────────────────────────────
pipeline:
	@echo "→ Running FULL pipeline (season=$(SEASON))..."
	mm run full --season $(SEASON)

pipeline-intel:
	@echo "→ Running INTEL update (season=$(SEASON))..."
	mm run intel --season $(SEASON)

pipeline-results:
	@echo "→ Running RESULTS ingestion (season=$(SEASON))..."
	mm run results --season $(SEASON)

# ── Testing ──────────────────────────────────────────────────────────────────
test: test-backend test-frontend

test-backend:
	@echo "→ Running pytest..."
	$(PYTHON) -m pytest tests/ -q --tb=short

test-frontend:
	@echo "→ Running vitest..."
	cd frontend && npx vitest run --reporter=dot

test-coverage:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80

# ── Data utilities ────────────────────────────────────────────────────────────
validate:
	@echo "→ Checking data source connectivity..."
	$(PYTHON) scripts/validate_scraping.py

cache:
	@echo "→ Pre-populating T-Rank cache for season $(SEASON)..."
	$(PYTHON) scripts/warm_cache.py --season $(SEASON)

# ── ML training ──────────────────────────────────────────────────────────────
train:
	$(PYTHON) scripts/run_pipeline.py --season $(SEASON) --sampler advi

sweep:
	$(PYTHON) scripts/sweep.py

# ── Build & clean ─────────────────────────────────────────────────────────────
build:
	cd frontend && npm run build

clean:
	rm -rf data/cache/*.parquet data/cache/*.json
	rm -rf frontend/.next
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	@echo "✅ Clean complete."
