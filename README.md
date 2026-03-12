# Ethereal Oracle — NCAA March Madness Engine

A professional-grade bracket prediction and sports betting syndicate engine combining a **Spatio-Temporal Graph Neural Network (ST-GNN)** with a **Bayesian inference head** to generate full probability distributions over game outcomes, identify Closing Line Value (CLV), and optimize bracket topology under uncertainty.

---

## Overview

| Layer | Purpose |
|---|---|
| **ST-GNN** | Models the NCAA season as a heterogeneous graph (Team + Conference nodes, Game edges) |
| **Bayesian Head** | PyMC posterior distributions over win probability and point spread |
| **Monte Carlo Engine** | Full 6-round bracket simulations with Chaos Engine disruption logic |
| **Betting Layer** | Fractional Kelly sizing, CLV computation, Black-Scholes options hedging |
| **War Room UI** | Next.js dashboard with 3D constellation view, interactive bracket, and matchup interrogator |

Primary targets: **Brier Score** and **Closing Line Value (CLV)**. All outputs are probability distributions, never scalar point estimates.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Data Ingestion                        │
│  Barttorvik · Sports Reference CBB · Kaggle · Rotowire   │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│              PyG Heterogeneous Graph                      │
│  Team Nodes ──member_of──▶ Conference Nodes              │
│  Game Edges: court location · rest · travel fatigue      │
│  GAT (spatial SoS) + LSTM/Transformer (temporal)         │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│             Bayesian Head (PyMC / NUTS)                   │
│  Skellam margin likelihood · coach ATS priors            │
│  Luck regression prior · clutch shrinkage                 │
│  Posterior: P(win) distribution + spread distribution    │
└─────────────────────────┬────────────────────────────────┘
                          │
          ┌───────────────┴────────────────┐
          │                                │
┌─────────▼──────────┐        ┌────────────▼───────────┐
│  Monte Carlo        │        │  Betting Layer          │
│  Bracket Engine     │        │  Kelly · CLV · Hedging  │
│  Chaos Engine       │        │  Options pricing (BSM)  │
│  Copula correlation │        │  Prospect theory        │
└─────────┬──────────┘        └────────────┬───────────┘
          │                                │
          └───────────────┬────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                          │
│  /api/graph · /api/matchup · /api/bracket/simulate       │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│               Next.js War Room UI                        │
│  3D Constellation · Interactive Bracket · Heatmap        │
│  Matchup Interrogator · WPA Sliders · Leverage Matrix    │
└──────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
| Component | Library |
|---|---|
| Graph Neural Network | `torch`, `torch_geometric` (PyG) |
| Bayesian Inference | `pymc`, `arviz` |
| API Server | `fastapi`, `uvicorn` |
| Data Wrangling | `pandas`, `polars`, `numpy`, `scipy` |
| Graph Construction | `networkx` |
| Web Scraping | `requests`, `httpx`, `beautifulsoup4`, `playwright` |
| MLOps | `wandb` |
| Testing | `pytest`, `pytest-cov` |

### Frontend
| Component | Library |
|---|---|
| Framework | Next.js 14 (App Router) |
| 3D Visualization | `@react-three/fiber`, `@react-three/drei`, `three` |
| Animations | `framer-motion` |
| Testing | `vitest`, `@testing-library/react` |
| E2E | `playwright` |

---

## Data Sources

All sources are **100% free** — no paid API keys required.

| Source | Data | Module |
|---|---|---|
| **Kaggle March Mania** | Historical results, seeds, spread archives | `src/data/kaggle_ingestion.py` |
| **Barttorvik (T-Rank)** | AdjO, AdjD, Tempo, Luck, PORPAGATU!, BPM, roster continuity | `src/data/barttorvik.py` (via `sports_reference.py`) |
| **Sports Reference CBB** | Four Factors, ORtg, DRtg, 3PA rate, FTA rate, coach records | `src/data/sports_reference.py` |
| **Rotowire CBB** | Injury / availability feed | `src/data/injury_feed.py` |
| **Sportsbook Review** | Historical opening/closing lines for CLV | `src/data/market_data.py` |
| **Yahoo / ESPN / NCAA** | Public pick percentages for leverage scoring | `src/data/public_picks.py` |

---

## Key Models

### Zero-Truncated Skellam Distribution
Replaces the Normal spread likelihood. Models margin-of-victory as the difference of two Poisson scoring processes, **guaranteeing P(tie) = 0** in accordance with NCAA overtime rules.

### Coach-Level Hierarchical Prior ("Tom Izzo Effect")
Partial pooling over head coaches — infers a per-coach `coach_ats_effect` that encodes systematic over/under-performance in tournament formats relative to regular-season efficiency metrics.

### Clutch/Luck Regression Prior
`pm.Beta` or `pm.Normal` prior centered at 0.5 (σ ≤ 0.15) on the luck parameter. A team going 10-0 in close games has its posterior penalized toward mean volatility, consistent with Law of Large Numbers regression over a 35-game sample.

### Chaos Engine
When a 1- or 2-seed is eliminated in Rounds 1–2, the bracket simulator reweights all surviving teams in the affected region against the revised field, applies fatigue penalties (`−0.02/game`, `−0.015` for OT), and resamples all subsequent pairwise probabilities from updated posteriors.

### Gaussian Copula Correlation
Replaces independent Monte Carlo draws. Conference-correlated draws propagate upsets — when a conference favorite falls, all co-conference teams receive a configurable contagion downgrade (default `−3%`).

---

## Project Structure

```
MM/
├── src/
│   ├── api/               # FastAPI server + schemas
│   ├── data/              # Ingestion modules (one per source)
│   ├── graph/             # PyG graph construction, node/edge features
│   ├── model/             # GAT encoder, temporal encoder, Bayesian head, Skellam
│   ├── simulation/        # Monte Carlo engine, Copula, RL bracket optimizer
│   ├── betting/           # Kelly sizing, CLV, Black-Scholes, Prospect Theory
│   ├── backtesting/       # Historical calibration + Brier score evaluation
│   └── pipeline/          # End-to-end pipeline runner
├── tests/                 # pytest suite mirroring src/ (1018 tests)
├── frontend/
│   ├── app/               # Next.js App Router pages
│   ├── components/
│   │   ├── bracket/       # Interactive 6-round bracket + heatmap
│   │   ├── constellation/ # 3D R3F scene (court, team/conference nodes, edges)
│   │   └── nav/           # Sidebar navigation
│   ├── lib/               # API client, types, mock data, bracket utils
│   └── e2e/               # Playwright end-to-end tests
├── docs/plans/            # Implementation plans
├── tasks/                 # todo.md + lessons.md session tracking
├── notebooks/             # EDA scratchpads
├── scripts/               # CLI pipeline runner
└── artifacts/             # W&B artifacts, saved models (gitignored)
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- Node.js 18+
- Kaggle account (for dataset downloads)
- Weights & Biases account (for experiment tracking)

### 1. Backend Setup

```bash
# Clone and install
git clone https://github.com/ritvikv03/MM.git
cd MM
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in: KAGGLE_USERNAME, KAGGLE_KEY, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY

# Run tests
python -m pytest tests/ -q
# Expected: 1018 passed

# Start API server
uvicorn src.api.server:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd frontend
npm install

# Start dev server
npm run dev
# Open http://localhost:3000

# Run unit tests
npx vitest run

# Run E2E tests (requires backend + dev server running)
npx playwright test
```

---

## Frontend Pages

| Page | Description |
|---|---|
| **Bracket** | Interactive 6-round bracket (R64 → Championship). Click any team to pick a winner; model cascades into downstream rounds automatically. Includes Monte Carlo simulation button and advancement heatmap. |
| **Matchup** | Matchup Interrogator — head-to-head win probability, spread distribution, KDE plot, WPA sliders for manual efficiency adjustment. |
| **Graph** | 3D constellation view — team and conference nodes floating above a Three.js basketball court. Orbit + zoom. Season selector triggers graph reload. |
| **Rankings** | Team efficiency rankings with Barttorvik-derived metrics. |
| **Projections** | Pre-tournament bracket projections with per-seed advancement probabilities. |
| **War Room** | Leverage matrix heat-map (Green = high value / low ownership, Red = toxic chalk). |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/graph?season=YYYY` | Team + conference graph with positions and edge data |
| `POST` | `/api/matchup` | Head-to-head win probability and spread distribution |
| `POST` | `/api/bracket/simulate` | Monte Carlo bracket simulation (n_simulations runs) |

---

## Test Suite

```bash
# Backend (Python)
python -m pytest tests/ -q                    # 1018 tests
python -m pytest tests/ --cov=src --cov-report=term-missing

# Frontend (Vitest)
cd frontend && npx vitest run

# E2E (Playwright)
cd frontend && npx playwright test
```

---

## MLOps

Every training run logs to Weights & Biases:

```python
wandb.log({
    "brier_score": ...,
    "log_loss": ...,
    "clv_delta": ...,
    "calibration_ece": ...,
    "epoch": ...,
})
```

Runs are tagged with `season=YYYY` and `model_version=vX.Y`.

---

## Development Phases

| Phase | Milestone | Status |
|---|---|---|
| 0 | Environment setup | ✅ |
| 1 | Data ingestion pipelines | ✅ |
| 2 | Graph constructor | ✅ |
| 3 | ST-GNN (GAT + temporal encoder) | ✅ |
| 4 | Bayesian head (PyMC) | ✅ |
| 5 | Monte Carlo bracket simulation + Chaos Engine | ✅ |
| 6 | Kelly sizing + CLV computation | ✅ |
| 7 | W&B experiment tracking | ✅ |
| 8 | Backtesting + calibration | ✅ |
| 9 | End-to-end pipeline + CLI | ✅ |
| 10 | War Room frontend | ✅ |

---

## License

MIT
