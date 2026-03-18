# CLAUDE.md — Master System Prompt & Workflow Guide
## Madness Matrix — Production-Deployed ST-GNN + Bayesian Bracket Prediction Engine

---

## 1. Primary Project Goal

Build **Madness Matrix** — a production-deployed, autonomously-running NCAA March Madness prediction engine that finds **Closing Line Value (CLV)** and maximizes **Brier Score**. The system runs 3× daily via GitHub Actions, stores every prediction snapshot in Supabase for historical accuracy tracking, and surfaces descriptive, predictive, and prescriptive analytics in a deployed Next.js frontend. The current target bracket is **2026**.

**Core Mandate:**
- Move entirely away from point-estimate tabular classification (e.g., scikit-learn / XGBoost pipelines).
- Leverage a **Spatio-Temporal Graph Neural Network (ST-GNN)** combined with **Bayesian Inference**.
- All model outputs must be **probability distributions** of game outcomes, not scalar win-probability estimates.
- The model optimizes for **two targets only**: binary win/loss probability and point spread (margin of victory). Game totals are excluded — see §2 Probabilistic Modeling for rationale.
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
- **PyTorch Geometric (PyG)** — model the NCAA season as a **Heterogeneous Graph** with two distinct node types
  - **Graph Attention Networks (GAT)** for spatial Strength-of-Schedule encoding
  - **LSTM / Transformer layers** for temporal momentum (rolling game-by-game sequences)
  - **Node Types (Heterogeneous Graph):**
    - **Team Nodes** — one per D-I program; features = efficiency metrics, availability, roster continuity
    - **Conference Nodes** — one per conference (ACC, Big 12, SEC, MAC, etc.); features = conference-level adjusted efficiency aggregates
    - Every Team Node must be connected to its primary Conference Node via a directed `member_of` edge. This allows the GAT to explicitly model inter-conference strength disparities (e.g., an average Big 12 team vs. a dominant MAC team) rather than treating all teams in a contextual silo.
  - Each game edge = directed Team→Team, weighted by margin/efficiency delta
  - **Edge Features:** Must explicitly encode:
    - **Court Location** (Home/Away/Neutral): Tournament games are played on neutral courts; the GAT must isolate neutral-court baseline strength from home-court inflation.
    - **Rest Disparity**: days since last game for both teams.
    - **Travel Fatigue / Altitude Delta (mandatory):** For each game edge, compute (a) **Distance Traveled** in miles (great-circle distance from home campus to tournament venue), (b) **Time Zones Crossed** (absolute difference in UTC offsets, discretized 0/1/2/3+), and (c) **Elevation Flag** (boolean; 1 if tournament venue elevation > 5,000 ft above sea level — e.g., Denver/Salt Lake City/Albuquerque). These three sub-features are concatenated into a `travel_fatigue` edge feature vector of shape (3,). Campus coordinates are pulled from a static lookup CSV; venue coordinates are scraped from NCAA bracket pairings. Teams crossing ≥3 time zones with ≤48 h rest receive the maximum fatigue penalty the GAT can learn to apply.

### Probabilistic Modeling
- **PyMC** (preferred) or **Stan** for Bayesian multi-task outcome generation
  - Posterior distributions over **win probability** and **point spread (margin of victory) only**
  - **Game Totals are permanently excluded from the model objective.** End-of-game fouling protocols introduce noise that has no mathematical bearing on team quality, Brier Score accuracy, or CLV. Do not add an `obs_total` likelihood term to the PyMC model. Do not predict, log, or backtest total points.
  - Hierarchical priors over conferences, seeds, and **coaches**
  - **Coach-Level Hierarchical Prior ("Tom Izzo Effect" — mandatory):** Inject a `coach_id` integer index (one per head coach) into the Bayesian model head.  For each coach, infer a latent `coach_ats_effect` variable:
    ```
    mu_coach   ~ Normal(0, 0.5)          # league-wide mean ATS tendency
    sigma_coach ~ HalfNormal(0.3)        # cross-coach variation
    coach_ats_effect[c] ~ Normal(mu_coach, sigma_coach)   # per-coach partial pooling
    ```
    The `coach_ats_effect` is added to `delta` (home_strength − away_strength) before the Bernoulli and Normal likelihoods.  This encodes the empirical observation that coaches like Tom Izzo (Michigan State), Bill Self (Kansas), and John Calipari (Kentucky) systematically over- or under-perform regular-season efficiency metrics in sudden-death tournament formats.  Historical coaching ATS data must be scraped from Sports Reference CBB coaching records.  Implemented in `bayesian_head.py::build_model(home_coach, away_coach, ...)` with shapes `(G,)` int arrays indexed into `coach_ats_effect`.
  - MCMC / NUTS sampler for credible intervals
  - **Clutch/Luck Regression Prior (mandatory):** The Bayesian head must encode an explicit shrinkage prior on "clutch" performance metrics sourced from Barttorvik's Luck metric and close-game win percentage (games decided by ≤3 points). Mathematically, over a 35-game NCAA regular season sample, close-game outcomes revert strongly to mean. Implement this as a `pm.Beta` or `pm.Normal` prior centered at 0.5 with a tight sigma (≤0.15) on the luck/clutch parameter, so that a team going 10-0 in close games has its posterior win-probability distribution penalized downward toward average volatility rather than being treated as a genuine skill signal. This prior must be documented in the model's docstring with the cite: "Law of Large Numbers regression over 35-game samples."
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
  - **Player BPM must be computed as a Rolling EWMA, not a season-long average.**  The pipeline uses `pandas.DataFrame.ewm(halflife=15, times=game_dates)` (or equivalent) to weight each player's per-game BPM contribution so that games in the final 15 days of the regular season carry the dominant weight.  This captures "Freshman Pop" and late-season rotation cohesion — a team that discovered its lineup in late February should be rated on its current performance gradient, not dragged down by early-season turnover.  Implemented in `barttorvik.py::compute_ewma_bpm(player_games_df, halflife_days=15)` returning a per-player weighted BPM Series, then aggregated by minutes-weighted mean to a team-level scalar.
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
├── .env.example                 # Template listing all required keys
├── .gitignore
│
├── .github/
│   └── workflows/
│       ├── daily_pipeline.yml   # 3× daily cron: 6 AM / 12 PM / 10 PM ET
│       └── pr_tests.yml         # CI: pytest + vitest on every PR
│
├── src/
│   ├── data/                    # All ingestion modules (one per source)
│   ├── graph/                   # PyG graph construction & temporal encoding
│   │   ├── graph_constructor.py
│   │   ├── node_features.py
│   │   └── edge_features.py
│   ├── model/                   # ST-GNN + Bayesian + calibration
│   │   ├── gat_encoder.py
│   │   ├── temporal_encoder.py
│   │   ├── bayesian_head.py
│   │   ├── calibration.py       # Isotonic regression calibration layer (NEW)
│   │   └── ensemble.py
│   ├── simulation/              # Monte Carlo bracket engine
│   │   └── monte_carlo.py
│   ├── betting/                 # Kelly Criterion sizing, CLV computation
│   │   └── kelly.py
│   ├── pipeline/                # GitHub Actions orchestration
│   │   ├── github_actions_runner.py  # full / intel / results run profiles
│   │   ├── supabase_writer.py        # writes all outputs to Supabase
│   │   └── config.py
│   └── utils/                   # Logging, config, shared helpers
│
├── supabase/
│   └── migrations/
│       └── 001_initial_schema.sql   # All 7 tables
│
├── frontend/                    # Next.js 15 (deployed on Vercel)
│   ├── app/
│   ├── components/
│   │   ├── live/                # Live 2026 tab (Descriptive/Predictive/Prescriptive)
│   │   ├── rankings/
│   │   ├── matchup/
│   │   ├── bracket/
│   │   ├── warroom/
│   │   ├── intel/               # Intel Feed (Supabase Realtime)
│   │   └── nav/
│   └── lib/
│       ├── supabase.ts          # Supabase client singleton
│       └── queries.ts           # React Query hooks
│
├── tests/                       # pytest test suite mirroring src/
├── docs/plans/                  # Design docs and implementation plans
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
- All keys must live in `.env` only (local) and GitHub Actions Secrets / Vercel Environment Variables (deployed). Never hardcoded.
- Load via `python-dotenv`. `.env` is gitignored; `.env.example` lists all required keys.
- **Required keys:**
  ```
  KAGGLE_USERNAME           # Kaggle data downloads
  KAGGLE_KEY
  WANDB_API_KEY             # W&B experiment tracking
  WANDB_PROJECT
  WANDB_ENTITY
  SUPABASE_URL              # Supabase project URL
  SUPABASE_SERVICE_KEY      # Service role key (server/Actions only — never in frontend)
  SUPABASE_ANON_KEY         # Anon key (safe for frontend/Vercel)
  VERCEL_CRON_SECRET        # Shared secret for cron webhook authentication
  ```
- No other API keys are needed or permitted. All scraped data comes from free public pages.

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
| 5 | Monte Carlo bracket simulation engine + Chaos Engine | `/using-git-worktrees` |
| 6 | Kelly Criterion sizing + CLV computation | `/using-git-worktrees` |
| **CHECKPOINT** | Code review of `src/betting/` | `/requesting-code-review` |
| 7 | W&B experiment tracking integration | — |
| 8 | Backtesting + calibration evaluation | `/subagent-driven-development` |

### Monte Carlo Simulation — "Chaos Engine" (Topology Disruption Rule)

The bracket simulator in `src/simulation/monte_carlo.py` must implement a **Topology Disruption Rule** that activates when a major-seed upset occurs:

**Trigger condition:** A 1-seed or 2-seed team is eliminated in Rounds 1 or 2 within a given bracket region.

**Disruption logic (applied per simulation trial):**
1. **Identify affected nodes:** All surviving teams in the same bracket region as the eliminated titan.
2. **Recompute path difficulty:** Without the titan, the expected difficulty of the remaining path drops. Reweight each surviving team's advancement probability using its posterior spread distribution against the revised field (remove the titan's influence from the denominator of path-probability products).
3. **Apply fatigue/momentum adjustment:** Model the psychological + physical wear-and-tear. For each game a team played to eliminate the titan, add a `chaos_fatigue_penalty` (default: `−0.02` to win probability per game played at max exertion, configurable). Surviving teams that played an overtime game inherit an additional `−0.015` penalty for the next round.
4. **Re-draw remaining bracket sub-tree:** For the affected region, resample all pairwise matchup probabilities using updated team-strength posteriors (with the titan's bracket path collapsed), then continue the simulation forward.

This ensures that when Kansas exits in Round 1, the simulator correctly re-evaluates the Midwest region's path as "open" rather than applying pre-computed single-elimination probabilities that assumed Kansas would reach the Elite Eight.

Implement as `_apply_chaos_engine(bracket_state, eliminated_team, region, posteriors, rng)` returning an updated `bracket_state` dict. Must be called inside the main simulation loop immediately after each round resolution.

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

## 8. Executive-Level Production Layer

To manage risk, optimize pool leverage, and provide a war-room dashboard for high-stakes betting syndicates, the architecture includes an Executive-Level Production Layer encompassing 5 distinct phases:

### Phase 1: Game Theory & Pool Optimization (The "Leverage" Engine)
- **Public Pick Scraper:** Ingests "Public Pick Percentages" from free sources (Yahoo/ESPN/NCAA).
- **Leverage Calculator:** Computes `Leverage Score = True_Win_Probability / Public_Pick_Percentage`.
- **Bracket Variants:** Automatically generates 3 distinct topologies:
  - **Chalk Bracket:** Optimized for small pools (<20 people).
  - **Leverage Bracket:** Optimized for medium pools (50-100 people) by fading over-owned favorites.
  - **Chaos Bracket:** Optimized for massive contests (1,000+ people) by prioritizing low-ownership deep runs.

### Phase 2: Live Dynamic Hedging (The "Risk Desk")
- **Live Hedging Calculator:** Utilizes fractional Kelly-Criterion sizing based on live Moneyline odds.
- **EV Lock:** Automatically calculates the exact dollar amount needed to bet on the opposing team to lock in a risk-free profit based on current pool standings and equity.

### Phase 3: The "Board of Directors" Ensemble Weighting
- Refactors the ST-GNN + Bayesian head as the primary engine but introduces three lightweight secondary voter models:
  - **The Fundamentalist:** Weights only AdjO/AdjD/Rebounding efficiency.
  - **The Market Reader:** Weights only historical spread movement and sharp money indicators.
  - **The Chaos Agent:** Weights officiating tendencies, travel fatigue, altitude, and momentum/kill-shot vulnerability.
- **Consensus Output:** Produces a final summary stating overall confidence and specific dissent reasons.

### Phase 4: Continuous "Information Asymmetry" Scraper
- Lightweight background task scraping subreddits (e.g., `r/CollegeBasketball`) and news feeds using free tools (`BeautifulSoup`, `Playwright`, `PRAW`).
- **Keyword Monitoring:** Alerts on terms like "walking boot," "sprain," "suspension," or "not at practice."
- Flags affected matchups for manual human review before committing the simulation path.

### Phase 5: The "War Room" UI
- Next.js dashboard integrating the backend metrics.
- **Matrix View:** Heat-map of the entire bracket colored by `Leverage Score` (Green = High Value/Low Ownership; Red = Toxic Chalk).
- **WPA (Win Probability Added) Slider:** Interactive slider in the Matchup Interrogator to manually adjust team efficiency (simulating injury/foul trouble) and instantly visualize the dynamic shift in bracket topology.

---

## 9. Boss-Level Mathematical Upgrades

Advanced mathematical, economic, and ML theories integrated into the core engine.

### Pillar 1: Advanced Statistical & Temporal Modeling

#### 1.1 Zero-Truncated Skellam Distribution (`src/model/skellam.py`)
- Replaces the Normal spread likelihood with a **Zero-Truncated Skellam** distribution.
- Models margin-of-victory as the difference of two Poisson scoring processes.
- **Guarantees P(tie) = 0**, matching NCAA overtime rules.
- Uses Modified Bessel functions (I_v) for exact PMF computation.
- References: Karlis & Ntzoufras (2003); Skellam (1946).

#### 1.2 Shannon Entropy Scoring Variance (`src/data/shannon_entropy.py`)
- Computes **minute-by-minute Shannon Entropy** (H = -Σ p log₂ p) to quantify offensive consistency.
- High entropy = consistent scoring; Low entropy = bursty "Kill Shot" tendencies.
- **Kill Shot Markov Matrix** (4-state): Dead Ball → Home Run → Away Run → Trading Baskets.
- Injected as node features into the ST-GNN alongside efficiency metrics.

#### 1.3 Gaussian Copula Cross-Game Correlation (`src/simulation/copula_engine.py`)
- Replaces independent Monte Carlo draws with **conference-correlated Gaussian Copula** draws.
- When a conference favorite is upset, all other teams from that conference receive a **contagion downgrade** (configurable, default −3%).
- Ensures the Chaos Engine captures real-world market correction dynamics.
- References: Nelsen (2006); McNeil, Frey, Embrechts (2005).

### Pillar 2: Financial Engineering & Game Theory

#### 2.1 Real Options Valuation / Black-Scholes (`src/betting/options_pricing.py`)
- Treats tournament advancement as a **European option chain** with per-round expiry.
- Computes **Vega** (∂C/∂σ = S√T φ(d1)) to quantify path volatility sensitivity.
- Hedge recommendations: High Vega + High Equity → HEDGE_AGGRESSIVELY; High Vega + Low Equity → LET_IT_RIDE.

#### 2.2 Prospect Theory CLV Identification (`src/betting/prospect_theory.py`)
- Implements the **Prelec (1998) probability weighting function**: w(p) = exp(−(−ln p)^α).
- Identifies the Favorite-Longshot Bias: public overweights small p by ~8%, underweights large p by ~10%.
- **CLV Scanner**: scans matchup slates and ranks opportunities by raw CLV magnitude.
- Peak irrationality defaults: 5-vs-12 seed matchups in R64.

#### 2.3 RL Bracket Optimization (`src/simulation/rl_bracket.py`)
- Lightweight RL pool environment simulating 1,000+ "Dumb Brackets" from public pick percentages.
- **Greedy Leverage Agent** selects picks that maximize Leverage Score while maintaining viability.
- Achieves **Top-10 rate ~48%, Win rate ~18%** against 100-entry pools in Monte Carlo testing.

---

*Last updated: 2026-03-12 — Full Madness Matrix overhaul: deployment architecture, no-synthetic-data mandate, 3× daily pipeline, §15–17 added.*

---

## 10. Agentic Workflow Rules & Developer Principles

### 10.1 Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions).
- If something goes sideways, STOP and re-plan immediately - don't keep pushing.
- Use plan mode for verification steps, not just building.
- Write detailed specs upfront to reduce ambiguity.

### 10.2 Subagent Strategy
- Use subagents liberally to keep main context window clean.
- Offload research, exploration, and parallel analysis to subagents.
- For complex problems, throw more compute at it via subagents.
- One task per subagent for focused execution.

### 10.3 Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern.
- Write rules for yourself that prevent the same mistake.
- Ruthlessly iterate on these lessons until mistake rate drops.
- Review lessons at session start for relevant project.

### 10.4 Verification Before Done
- Never mark a task complete without proving it works.
- Diff behavior between main and your changes when relevant.
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness.

### 10.5 Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer.
- Challenge your own work before presenting it.

### 10.6 Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding.
- Point at logs, errors, failing tests - then resolve them.
- Zero context switching required from the user.
- Go fix failing CI tests without being told how.

## 11. Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items.
2. **Verify Plan**: Check in before starting implementation.
3. **Track Progress**: Mark items complete as you go.
4. **Explain Changes**: High-level summary at each step.
5. **Document Results**: Add review section to `tasks/todo.md`.
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections.

## 12. Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.

---

## 13. Frontend Codebase Notes (React / Three.js / Next.js)

- `parseInt("55-23")` → `55` (wins only). Use `parseInt(s.split('-')[0], 10) || 0` for W-L record strings.
- `new THREE.Color(node.color)` fails with raw int — always use `` `#${n.toString(16).padStart(6,'0')}` ``.
- Wrap `new THREE.Color()` / `new THREE.Vector3()` in `useMemo([dep])` inside R3F components — instantiating in render body causes per-frame material re-uploads.
- React Rules of Hooks: `useMemo`/`useCallback`/`useEffect` must appear **before** any `if (!x) return null` guard.
- R3F disposal: `useEffect` cleanup must call `.dispose()` on texture **and** geometry **and** material separately.
- `useFrame` with 360+ nodes: gate rotation/animation on `hovered || isSelected` — unconditional `useFrame` on all nodes saturates the JS thread at ~23k mutations/sec.
- `AnimatePresence mode="wait"` is required on panel sidebars — without it, exit and enter animations overlap producing ghost panels.
- `probToHeatColor` must return `rgb(r,g,b)` (no spaces, no alpha channel) — vitest regex `^rgb\(\d+,\d+,\d+\)$` tests this.
- R3F edge arrays: use content-based keys (`source-target`) not array index — index keys cause full re-mounts when graph data refreshes.
- Frontend vitest: ~5s. Backend pytest: ~20s. Run `npx vitest run --reporter=dot` for fast iteration.

---

## 15. Deployment Architecture

### Stack (all free, zero cost)
| Layer | Service | Purpose |
|---|---|---|
| Pipeline | GitHub Actions (public repo) | 3× daily cron: 6 AM / 12 PM / 10 PM ET. Runs scrape + ML + Monte Carlo. Writes to Supabase. |
| Database | Supabase (free project) | Postgres + Realtime. 7 tables. 500MB free tier. |
| Frontend | Vercel (Hobby plan) | Next.js 15 deployed globally. Reads Supabase directly. |
| Repo | github.com/ritvikv03/MM | Public. Unlimited Actions minutes. |

### Cron Schedule
```yaml
# .github/workflows/daily_pipeline.yml
schedule:
  - cron: '0 11 * * *'   # 6 AM ET  — full: scrape all + train + predict + write
  - cron: '0 17 * * *'   # 12 PM ET — intel: injuries/alerts + warm-start update
  - cron: '0 3 * * *'    # 10 PM ET — results: ingest outcomes + Brier score update
```

### Run Profiles
- **`full`** (6 AM): All scrapers → ST-GNN → Bayesian ADVI → Calibration → MC 10k → CLV → RL → Supabase
- **`intel`** (12 PM): Intel scrapers only → severity scoring → temporal Bayesian warm-start → partial snapshot update
- **`results`** (10 PM): Kaggle game results → Brier/Log-Loss computation → isotonic recalibration trigger

### No Persistent Backend Server
There is no always-on backend process (no Fly.io, no Railway, no Heroku).
- Heavy ML runs in GitHub Actions (7GB RAM, 2 cores, free)
- Frontend reads pre-computed predictions from Supabase via Next.js API routes
- On-demand matchup calculations use a lightweight analytical logistic model (no PyMC at request time)
- Intel alerts push via Supabase Realtime (WebSocket, no polling needed)

---

## 16. No Synthetic Data — Hard Rule

**SYNTHETIC DATA IS PERMANENTLY FORBIDDEN in Madness Matrix.**

This rule overrides all fallbacks, convenience stubs, and demo modes.

1. **Delete all stub functions** — `_build_stub_graph()`, `_build_stub_matchup()`, `_build_stub_simulate()` and all `_build_stub_*` variants are removed from `server.py`. Do not recreate them.
2. **Delete `frontend/lib/mock-data.ts`** — this file must not exist. Do not recreate it.
3. **Delete `USE_REAL_DATA` env var** — the flag and all `os.getenv("USE_REAL_DATA")` checks are removed. Real data is always used.
4. **Delete `StubDataBanner`** — the yellow warning banner component is removed. No stub mode exists.
5. **If a data source fails:** Return HTTP 503 with a clear error message. Never fabricate data as a fallback.
6. **If Supabase is unreachable:** Return 503. Never serve synthetic predictions.
7. **All predictions carry uncertainty.** Point estimates alone are forbidden. Every prediction must include posterior credible intervals or confidence bands.

**Rationale:** Synthetic data misleads users into making real bracket decisions based on random numbers. The entire value proposition of Madness Matrix is rigorous quantitative forecasting — stub data destroys that completely.

---

## 17. Frontend Stack Standards

### Tech Stack (current — overrides §13 Three.js notes)
```
Framework:   Next.js 15 (App Router)
Styling:     Tailwind CSS + shadcn/ui
State:       @tanstack/react-query (server state), React useState (UI state)
Database:    @supabase/supabase-js (reads + Realtime subscriptions)
Charts:      Recharts (heatmaps, sparklines, calibration curves)
Animation:   Framer Motion (page transitions)
Validation:  Zod (all API response schemas)
Dates:       date-fns
Testing:     Vitest + @testing-library/react + Playwright
```

### Removed (do not use or reinstall)
- `three`, `@react-three/fiber`, `@react-three/drei` — Three.js/R3F removed with graph tab
- Inline `style={{}}` objects — use Tailwind classes instead
- Custom `GlassCard`, `GlowButton` — use shadcn/ui `Card`, `Button`
- Raw D3 in JSX — use Recharts; raw D3 only acceptable for KDE computation (`lib/d3-kde.ts`)

### Navigation (5 tabs — graph tab permanently removed)
```
[Live 2026] [Rankings] [Matchup] [Bracket] [War Room]
```
Season selector is **global state** — switching season updates ALL tabs simultaneously.

### Season Behavior
- **2026 (current):** Live tab enabled with real-time predictions. All tabs show live data.
- **2012–2025 (historical):** All tabs show frozen pre-tournament snapshot. Actual results overlaid as ✓/✗ for accuracy review. Live tab shows historical descriptive view.

### Live 2026 Tab — Three Sub-Views
- **Descriptive:** What happened — efficiency trends, results feed, injury timeline
- **Predictive:** What will happen — bracket probabilities, probability drift chart (temporal Bayesian), calibration curve (isotonic)
- **Prescriptive:** What to do — CLV picks, RL bracket variants, Kelly sizing, Intel alerts

### ML Outputs Surfaced in Frontend
All three new ML additions from the prediction engine must be visible in the UI:
1. **Isotonic calibration** → calibration curve in Live → Predictive sub-view (Recharts LineChart)
2. **Temporal Bayesian updating** → probability drift sparklines in Live → Predictive + Rankings trend column
3. **Conference RPI weighting** → star-tier badge in Rankings table + Chaos Agent panel in Matchup Oracle

---

## 14. Brand & Terminology Standards ("Madness Matrix")

**Brand Name:** Madness Matrix

**Official terminology mappings** — apply in all UI copy, component labels, and user-facing strings:

| Old term | Canonical replacement | Notes |
|---|---|---|
| "Pool" / "pools" | **"bracket contest"** or **"the tournament field"** | A "pool" is jargon; normal users submit a bracket to the NCAA or compete with friends. Use "bracket contest" for competitive groups, "tournament field" for the 68-team set. |
| "EV" | **"Alpha Rating"** or **"Upset Edge"** | Expected Value is quant jargon. Use "Alpha Rating" for model edge score, "Upset Edge" for matchup-specific upset opportunity. |
| "Betting" | **"Bracket Portfolio Strategy"** | |
| "Bracket Creator" | **"Bracket Architect"** | |

**In-code rules:**
- Backend mathematical code (`kelly.py`, `leverage.py`, `hedging.py`, `rl_bracket.py`) may retain `EV`, `pool`, `BracketPoolEnvironment` as technical identifiers — these are internal APIs.
- All user-visible strings in `frontend/` and model recommendations in `src/simulation/` must use the canonical replacements above.
- When referring to the 68-team NCAA field as a whole, use "The Tournament Field." When referring to a group of friends or contest entrants submitting brackets, use "bracket contest."
