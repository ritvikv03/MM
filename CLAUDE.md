# CLAUDE.md — Master System Prompt & Workflow Guide
## NCAA March Madness ST-GNN + Bayesian CLV Model

---

## 1. Primary Project Goal

Build a novel predictive model to find **Closing Line Value (CLV)** and maximize **Brier Score** for NCAA March Madness.

**Core Mandate:**
- Move entirely away from point-estimate tabular classification (e.g., scikit-learn / XGBoost pipelines).
- Leverage a **Spatio-Temporal Graph Neural Network (ST-GNN)** combined with **Bayesian Inference**.
- All model outputs must be **probability distributions** of game outcomes, not scalar win-probability estimates.
- Distributions feed downstream into:
  - **Monte Carlo Bracket Simulations** (full bracket path probability)
  - **Kelly Criterion betting** (fractional Kelly sizing from edge + variance estimates)

**Success Metrics (in priority order):**
1. Brier Score (primary)
2. Log-Loss
3. Closing Line Value (CLV) — model opening line vs. market close
4. ROI under fractional Kelly staking

---

## 2. Technology Stack

### Graph Processing
- **PyTorch Geometric (PyG)** — model the NCAA season as a dynamic directed graph
  - **Graph Attention Networks (GAT)** for spatial Strength-of-Schedule encoding
  - **LSTM / Transformer layers** for temporal momentum (rolling game-by-game sequences)
  - Each node = a team; each edge = a game played (directed, weighted by margin/efficiency delta)
  - **Edge Features:** Must explicitly encode Court Location (Home/Away/Neutral) and Rest Disparity (days since last game for both teams). Tournament games are played on neutral courts; the GAT must be able to isolate neutral-court baseline strength from home-court inflation.

### Probabilistic Modeling
- **PyMC** (preferred) or **Stan** for Bayesian multi-task outcome generation
  - Posterior distributions over win probability, point spread, and total
  - Hierarchical priors over conferences and seeds
  - MCMC / NUTS sampler for credible intervals
  - **Compute Optimization:** For local TDD and prototyping, default PyMC to Variational Inference (ADVI) or limit MCMC chains/draws to prevent hanging. Full NUTS sampling and PyG training sweeps should be configurable via CLI flags to run on cloud/GPU infrastructure when ready.

### MLOps Tracker
- **Weights & Biases (W&B)**
  - Log custom metrics: Brier Score, Log-Loss, CLV delta, calibration curves
  - Hyperparameter sweep grids (learning rate, GAT heads, LSTM hidden size, MCMC chains)
  - Track loss curves and model artifacts per experiment run

### Supporting Libraries
| Purpose | Library |
|---|---|
| Data wrangling | `pandas`, `polars` |
| Numerical ops | `numpy`, `scipy` |
| Graph ops | `networkx` (construction), `torch_geometric` (training) |
| Bayesian | `pymc`, `arviz` |
| HTTP scraping | `requests`, `httpx`, `playwright` |
| Scheduling | `APScheduler` |
| Testing | `pytest`, `pytest-cov` |
| Environment | `python-dotenv` |

---

## 3. Mandatory Data Sources

**All data sources are 100% free.** No paid subscriptions or premium API keys are permitted at any point in this pipeline. Each source maps to a dedicated ingestion module under `src/data/`.

### 3.1 Core Structure
- **Kaggle March Mania datasets** (free, requires Kaggle account)
  - Historical game-by-game results (regular season + tournament)
  - Tournament seeds (1985–present)
  - Team spellings / ID crosswalk
  - Historical NCAAB point spreads and totals archives (used for CLV computation — see §3.5)
  - Module: `src/data/kaggle_ingestion.py`

### 3.2 Advanced Team Efficiency
- **sportsipy** (Sports Reference CBB wrapper — free, no API key)
  - Four Factors, ORtg, DRtg, Pace, SOS
  - Module: `src/data/sports_reference.py`
- **Barttorvik (T-Rank)** — **sole source for AdjO, AdjD, Tempo, and luck-adjusted metrics**
  - Scrape `barttorvik.com` directly via BeautifulSoup/requests (public, free)
  - Adjusted efficiency margins, luck-adjusted ratings, pre-tournament projections
  - **AdjO, AdjD, Tempo, Luck** — replaces KenPom entirely; Barttorvik provides equivalent metrics at no cost
  - **PORPAGATU! (Points Over Replacement Per Adjusted Game At The Unit)** table — scrape from Barttorvik player pages to measure rotation strength and quantify injury impact on team nodes
  - **BPM (Box Plus-Minus)** table — scrape from Barttorvik player pages; replaces EvanMiya BPR entirely
  - **Roster Continuity / Returning Possession %** — scrape to dynamically widen Bayesian uncertainty priors for teams with high turnover early in the season
  - **KenPom is permanently removed.** Do not create or maintain `src/data/kenpom.py`. Do not reference KenPom credentials anywhere in the codebase.
  - Module: `src/data/barttorvik.py`

### 3.3 Player-Level Granularity
- **Barttorvik Player Data** — replaces EvanMiya entirely (see §3.2 above)
  - PORPAGATU! and BPM tables scraped directly from `barttorvik.com`
  - **EvanMiya is permanently removed.** Do not create or maintain `src/data/evanmiya.py`.
- **Sports Reference CBB** (free, public — replaces defunct Hoop-Math)
  - `hoop-math.com` domain is dead (DNS does not resolve as of 2026-03-10). **Do not attempt to scrape it.**
  - Use `sports-reference.com/cbb/seasons/men/{year}-school-stats.html` for shot-type proxies
  - Scrape: 3PA rate (3PA/FGA), FTA rate (FTA/FGA), eFG% — these serve as rim-rate, perimeter, and quality proxies
  - **3PA rate** = inverse proxy for rim/mid-range tendency (low 3PA rate → more interior shots)
  - **FTA rate** = proxy for attacking the rim (high FTA rate → high rim-attempt tendency)
  - **ShotQuality.com is permanently removed.** Do not create or maintain `src/data/shotquality.py`.
  - Module: `src/data/hoopmath.py` (rewritten to scrape Sports Reference)

### 3.4 Injury / Roster Availability
- **Rotowire CBB** injury pages — **sole injury source**
  - Scrape `rotowire.com/basketball/ncaa-injuries.php` via BeautifulSoup or Playwright
  - Dynamically downgrade team node strength when key players are ruled out
  - **Twitter/X API is permanently removed.** No social media fallback. Rotowire is authoritative.
  - Module: `src/data/injury_feed.py`

### 3.5 Market Data (CLV Computation)
- **Kaggle historical spread/totals archives** (primary) — free datasets containing NCAAB historical opening and closing lines
- **Sportsbook Review (SBR) public archives** — parse historical NCAAB lines from `sportsbookreview.com/betting-odds/college-basketball/` via BeautifulSoup/Playwright as a secondary source
  - CLV must still be computed relative to **sharp closing lines only** (Pinnacle, Circa, Bookmaker). Where historical sharp-book lines are unavailable, use the SBR consensus close as a proxy and flag the record.
  - **Public ticket/money splits are not tracked** unless a free data source becomes available.
  - **The Odds API is permanently removed.** Do not create or maintain any Odds API client.
  - **Action Network API is permanently removed.** Do not reference Action Network credentials anywhere.
  - Module: `src/data/market_data.py`

---

## 4. Project Directory Structure

```
MM/
├── CLAUDE.md                    # This file — master rulebook
├── README.md
├── pyproject.toml / requirements.txt
├── .env                         # API keys (never committed)
├── .gitignore
│
├── src/
│   ├── data/                    # All ingestion modules (one per source above)
│   ├── graph/                   # PyG graph construction & temporal encoding
│   │   ├── graph_constructor.py
│   │   ├── node_features.py
│   │   └── edge_features.py
│   ├── model/                   # ST-GNN + Bayesian model definitions
│   │   ├── gat_encoder.py
│   │   ├── temporal_encoder.py
│   │   └── bayesian_head.py
│   ├── simulation/              # Monte Carlo bracket engine
│   │   └── monte_carlo.py
│   ├── betting/                 # Kelly Criterion sizing, CLV computation
│   │   └── kelly.py
│   └── utils/                   # Logging, config, shared helpers
│
├── tests/                       # pytest test suite mirroring src/
│   ├── data/
│   ├── graph/
│   ├── model/
│   └── simulation/
│
├── notebooks/                   # EDA and research scratchpads only
└── artifacts/                   # W&B artifacts, saved models (gitignored)
```

---

## 5. Agentic Workflow Rules

### 5.1 Test-Driven Development (TDD)
- **All** data fetching, graph construction, and metric logging functions must follow **RED → GREEN → REFACTOR**.
- Write a failing `pytest` test first; implement only enough code to pass it; then refactor.
- Use the `/test-driven-development` skill for all new modules.
- Minimum coverage threshold: **80%** per module before merging.

### 5.2 Worktrees & Branch Isolation
- Major features must be developed in isolated git worktrees/branches.
- Use the `/using-git-worktrees` skill when starting any of these milestones:
  - `feature/graph-constructor`
  - `feature/bayesian-head`
  - `feature/monte-carlo-simulation`
  - `feature/kelly-betting`
  - `feature/data-<source-name>` (one branch per data source)
- Never commit directly to `main` during active development.

### 5.3 Subagent-Driven Parallelism
- Complex scraping and modeling tasks must be broken into **2–5 minute atomic tasks**.
- Use `/writing-plans` to decompose milestones before execution.
- Use `/subagent-driven-development` to execute tasks autonomously and in parallel wherever data sources are independent (e.g., Barttorvik + KenPom + ShotQuality can be scraped concurrently).

### 5.4 Mandatory Checkpoint Reviews
- The agent **must pause** and invoke `/requesting-code-review` after completing:
  1. All Graph Data constructors (`src/graph/`) — before Model Training begins
  2. Bayesian model head — before Monte Carlo integration
  3. Kelly sizing module — before any live market data is consumed
- No forward progress past a checkpoint without explicit user sign-off.

### 5.5 Secrets & API Key Management
- Only two services require credentials: **Kaggle** and **Weights & Biases**.
- All keys must live in `.env` only — never hardcoded in source files.
- Load via `python-dotenv`. `.env` is gitignored; `.env.example` lists the two required keys.
- Required: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`.
- No other API keys are needed or permitted. All other data is scraped from free public pages.

### 5.6 W&B Logging Standards
Every training run must log:
```python
wandb.log({
    "brier_score": ...,
    "log_loss": ...,
    "clv_delta": ...,
    "calibration_ece": ...,
    "epoch": ...,
})
```
Tag each run with `season=YYYY` and `model_version=vX.Y`.

---

## 6. Development Phases (Ordered)

| Phase | Milestone | Skill(s) |
|---|---|---|
| 0 | Environment setup, dependency install, `.env` config | — |
| 1 | Data ingestion pipelines (Kaggle, sportsipy, Barttorvik, Hoop-Math, Rotowire, SBR/Kaggle market) | `/test-driven-development`, `/subagent-driven-development` |
| 2 | Graph constructor — nodes, edges, temporal snapshots | `/using-git-worktrees`, `/test-driven-development` |
| **CHECKPOINT** | Code review of all `src/graph/` modules | `/requesting-code-review` |
| 3 | ST-GNN model — GAT encoder + LSTM/Transformer temporal | `/using-git-worktrees` |
| 4 | Bayesian head — PyMC posterior distributions | `/using-git-worktrees` |
| **CHECKPOINT** | Code review of `src/model/` | `/requesting-code-review` |
| 5 | Monte Carlo bracket simulation engine | `/using-git-worktrees` |
| 6 | Kelly Criterion sizing + CLV computation | `/using-git-worktrees` |
| **CHECKPOINT** | Code review of `src/betting/` | `/requesting-code-review` |
| 7 | W&B experiment tracking integration | — |
| 8 | Backtesting + calibration evaluation | `/subagent-driven-development` |

---

## 7. Constraints & Hard Rules

1. **Strict Point-in-Time (PIT) Integrity:** The data pipeline must never leak future data into past predictions. All historical training rows must use metrics (KenPom, BPR, Elo) exactly as they stood on the morning of the game. If daily historical snapshots are unavailable, the agent must construct a causal rolling calculation.
2. **No scikit-learn classifiers** as primary models. Utility functions (scalers, metrics) are acceptable.
3. **No point estimates as final output.** All predictions must carry uncertainty (posterior distribution or confidence interval).
4. **No live betting automation** without explicit user confirmation at each step.
5. **Respect rate limits** on all scraped sources — use exponential backoff and caching (cache raw responses to disk before parsing).
6. **Reproducibility** — set all random seeds (`torch`, `numpy`, `pymc`) and log them to W&B.
7. The agent must **never skip a checkpoint** even if the user requests speed. Flag the skip request and confirm explicitly.

---

*Last updated: 2026-03-10 — free-tier-only mandate applied; KenPom, EvanMiya, ShotQuality, Twitter/X, The Odds API, and Action Network permanently removed.*
