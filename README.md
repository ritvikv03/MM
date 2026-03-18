# Madness Matrix

**A production-grade NCAA March Madness bracket prediction engine powered by a Spatio-Temporal Graph Neural Network and Bayesian inference — running autonomously 3× daily via GitHub Actions.**

---

## The Problem

Every March, over 70 million brackets are filled out across the United States — and nearly all of them rely on gut instinct, seed numbers, and recency bias. The result is a market flooded with inefficiency.

Traditional bracket prediction approaches suffer from three fundamental flaws:

1. **Point estimates masquerade as certainty.** A model that says "Duke wins with 73% probability" communicates nothing about variance. Does that 73% have a credible interval of ±2% or ±18%? The answer changes everything about how to build a bracket.

2. **They treat games as independent events.** A standard classifier predicts each game in isolation. It cannot model the ripple effect when a 1-seed collapses in Round 1 — the path for every surviving team in that region just changed dramatically, and no logistic regression knows that.

3. **They ignore the market.** Bracket contests are not about picking the most likely outcome — they're about finding *mispriced* outcomes. A team with a 40% true win probability who appears on only 8% of public brackets has enormous leverage value, even if they're the underdog.

---

## The Solution

Madness Matrix is a full-stack quantitative forecasting system that treats bracket prediction as a **financial engineering problem** rather than a classification task.

The core thesis: **win probability distributions, not point estimates.** Every prediction the system produces is a posterior distribution — a full picture of uncertainty, not a single number. Those distributions feed into Monte Carlo bracket simulations, Kelly Criterion sizing, and a leverage engine that identifies where the public is wrong by the widest margin.

The result is a system that doesn't just predict who wins — it tells you *how confident to be*, *where the market is mispriced*, and *which bracket topology maximizes your edge* in a contest of any size.

---

## Why I Built This

March Madness bracket contests are one of the few remaining environments where a rigorous quantitative model has a persistent edge over the field. Unlike sports betting markets, which are efficient by design, public bracket contests are driven by narrative, fan loyalty, and television coverage. The information asymmetry is enormous.

I built Madness Matrix to answer a specific question: **what does a bracket look like when every pick is derived from a probability distribution rather than a heuristic?**

The secondary motivation was architectural. Most sports prediction pipelines are pipelines in name only — a sequence of Jupyter notebooks that produce a CSV. I wanted to build something that runs in production: ingesting live data, retraining on new information, writing predictions to a database, and surfacing the results in a deployed UI. The kind of system a quantitative analyst would actually ship.

---

## Methodology

### 1. Data Ingestion

All data sources are **100% free and public** — no paid API keys.

| Source | What It Provides |
|---|---|
| **Barttorvik (T-Rank)** | Adjusted Offensive/Defensive Efficiency, Tempo, Luck, PORPAGATU! player ratings, BPM, Roster Continuity |
| **Sports Reference CBB** | Four Factors, shot-type proxies (3PA rate, FTA rate, eFG%), historical coach records |
| **Kaggle March Mania** | Historical game-by-game results, tournament seeds, spread archives |
| **Rotowire CBB** | Injury and availability feed — sole injury source |
| **Yahoo / ESPN / NCAA** | Public pick percentages — used for leverage scoring |

Barttorvik is the primary efficiency source. KenPom is not used. All metrics are scraped in real time.

### 2. Graph Construction

The NCAA season is modeled as a **heterogeneous directed graph** using PyTorch Geometric:

- **Team Nodes** — one per D-I program. Features: efficiency margins, tempo, luck score, roster continuity, injury-adjusted strength
- **Conference Nodes** — one per conference. Features: conference-level RPI, aggregate efficiency, tier assignment
- **Game Edges** — directed, weighted by margin and efficiency delta. Edge features: court location (home/away/neutral), rest disparity, travel distance, time zones crossed, altitude flag
- **`member_of` Edges** — connects every Team Node to its Conference Node, allowing the model to explicitly encode inter-conference strength disparities

This structure allows an average Big 12 team to be correctly contextualized against a dominant MAC team — something tabular models fundamentally cannot do.

### 3. Model Architecture

**Spatial Layer — Graph Attention Network (GAT)**
The GAT encoder processes the full team graph and produces strength embeddings that account for strength-of-schedule, conference context, and edge features. Multi-head attention learns to weight different relationships differently.

**Temporal Layer — LSTM / Transformer**
Rolling game sequences capture momentum, hot/cold streaks, and rotation cohesion. Player BPM is computed as a rolling EWMA (halflife=15 days) — a team that found its lineup in late February is rated on its current form, not dragged down by early-season turnover.

**Bayesian Head (PyMC)**
The GAT embeddings feed into a PyMC model that infers posterior distributions over:
- **Win probability** — Bernoulli likelihood
- **Point spread** — Zero-Truncated Skellam distribution (difference of two Poisson processes; guarantees P(tie) = 0 per NCAA overtime rules)

Key priors:
- **Coach-level hierarchical prior** ("Tom Izzo Effect"): partial pooling over head coaches infers a per-coach `coach_ats_effect`. Coaches like Izzo and Bill Self systematically over/under-perform regular-season efficiency in sudden-death formats. This prior encodes that signal while sharing strength across coaches with limited tournament history.
- **Luck regression prior**: A tight prior (σ ≤ 0.15) on the luck parameter penalizes teams with extreme close-game records toward mean volatility — consistent with Law of Large Numbers regression over a 35-game sample. A team going 10-0 in overtime is probably lucky, not clutch.
- **Isotonic calibration**: Posterior probabilities pass through an isotonic regression calibrator before downstream simulation.

### 4. Monte Carlo Bracket Simulation

10,000 bracket trials per run. Each trial samples win probabilities from calibrated posteriors (Gaussian Copula draws — conference-correlated, not independent) and simulates all 63 games in sequence.

**Chaos Engine**: When a 1- or 2-seed is eliminated in Rounds 1–2, the simulator:
- Reweights all surviving teams in the affected region against the revised field
- Applies fatigue penalties (−0.02/game at max exertion; −0.015 for overtime games)
- Resamples all subsequent matchup probabilities from updated posteriors

### 5. Leverage Engine

- **Leverage Score** = `True Win Probability / Public Pick Percentage`
- Teams with Leverage Score > 1.5 are underowned relative to true probability — positive value picks
- Generates three bracket topologies: **Chalk** (small contests), **Leverage** (medium contests, fading over-owned favorites), **Chaos** (large contests, prioritizing low-ownership deep runs)
- **Prospect Theory CLV Scanner**: Implements the Prelec (1998) probability weighting function to identify maximum public mispricing (peak inefficiency at 5-vs-12 seed matchups)

### 6. Pipeline Automation

Three daily runs via GitHub Actions — no persistent server required:

| Time | Profile | What Runs |
|---|---|---|
| 6 AM ET | `full` | All scrapers → ST-GNN → Bayesian ADVI → Calibration → MC 10k → CLV → Supabase |
| 12 PM ET | `intel` | Injury/news scrapers → severity scoring → warm-start Bayesian update |
| 10 PM ET | `results` | Game results ingestion → Brier/Log-Loss update → recalibration trigger |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Data Ingestion Layer                                         │
│  Barttorvik · Sports Reference · Kaggle · Rotowire · ESPN    │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  PyG Heterogeneous Graph                                      │
│  Team Nodes ──member_of──▶ Conference Nodes                  │
│  Game Edges: court · rest · travel · altitude                │
│  GAT (spatial SoS) + LSTM/Transformer (temporal momentum)    │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  Bayesian Head (PyMC)                                         │
│  Skellam margin likelihood · Coach ATS priors                │
│  Luck regression shrinkage · Isotonic calibration            │
│  Output: P(win) posterior + spread posterior                 │
└──────────────┬─────────────────────────┬─────────────────────┘
               │                         │
┌──────────────▼──────────┐  ┌───────────▼──────────────────┐
│  Monte Carlo Engine      │  │  Betting / Leverage Layer     │
│  Chaos Engine            │  │  Kelly sizing · CLV scanner  │
│  Copula correlation      │  │  Prospect Theory · BSM hedge │
└──────────────┬──────────┘  └───────────┬──────────────────┘
               └─────────────┬───────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  FastAPI  →  Supabase  →  Next.js 15 (Vercel)                │
│  /api/intel · /api/matchup · /api/bracket/optimal            │
│  Live 2026 · Rankings · Matchup Oracle · Bracket · War Room  │
└──────────────────────────────────────────────────────────────┘
```

---

## Deployment Stack

| Layer | Service | Cost |
|---|---|---|
| Pipeline | GitHub Actions (public repo) | Free — unlimited minutes |
| Database | Supabase (Postgres + Realtime) | Free tier (500MB) |
| Frontend | Vercel (Next.js 15) | Free Hobby plan |
| Experiment Tracking | Weights & Biases | Free tier |

**Total infrastructure cost: $0.**

---

## Frontend

Five-tab dashboard:

| Tab | Description |
|---|---|
| **Live 2026** | Descriptive (efficiency trends, injury timeline) · Predictive (bracket probabilities, calibration curve, probability drift) · Prescriptive (CLV picks, Kelly sizing, Intel alerts) |
| **Rankings** | Power rankings with Barttorvik metrics, Conference RPI tier badges, temporal Bayesian probability drift sparklines |
| **Matchup Oracle** | Head-to-head analysis — win probability distribution, spread KDE, adjustable stat-weight factors (AdjOE, AdjDE, Luck, SOS, Coach, Tempo), coach matchup |
| **Bracket Architect** | Interactive 6-round bracket. Stat-weight toggle buttons. "Optimal Bracket" button auto-fills all picks from the 10k Monte Carlo simulation |
| **War Room** | Leverage matrix heatmap — Green = high true probability / low public ownership. Red = toxic chalk. |

---

## Limitations

**This system is as good as the data it ingests — no more, no less.**

1. **No live in-game data.** The pipeline runs 3× daily. If a star player exits injured at 11 AM, the prediction is stale until the 12 PM intel run.

2. **ADVI vs. NUTS.** Production uses variational inference (ADVI) as a fast approximation. ADVI is less accurate than full NUTS sampling for multimodal posteriors — a deliberate trade-off favoring update frequency.

3. **First-year coaches have no prior.** Coaches without historical tournament data default to the league-wide mean `coach_ats_effect`.

4. **Luck regression is conservative by design.** The tight prior will systematically underrate a genuinely clutch team if one exists. The model assumes clutch performance is luck until a large sample proves otherwise.

5. **Public pick percentage data is a snapshot.** Leverage scores are computed at a point in time. A major upset announcement shifts the public distribution faster than the pipeline re-ingests it.

6. **Pre-2018 calibration is thinner.** Barttorvik's full efficiency database is richest from 2018 onward. Earlier seasons use a reduced feature set.

7. **Not financial advice.** Kelly Criterion outputs are mathematical recommendations under a specific probability model. Fractional Kelly (0.25× full Kelly) is the default for a reason — variance in a 6-round single-elimination tournament is enormous.

---

## Outcomes and Evaluation

Primary metrics:
- **Brier Score** — mean squared error between predicted probability and actual outcome. Target: < 0.20 on tournament games.
- **Closing Line Value (CLV)** — spread between the model's opening prediction and the sharp market's closing line. Positive CLV over a large sample confirms genuine predictive edge.

Secondary:
- Log-Loss (penalizes confident wrong predictions more heavily)
- Calibration ECE (does "70% confident" win 70% of the time?)
- ROI under fractional Kelly in historical backtests

Every pipeline run writes Brier Score updates to Supabase as tournament results are ingested. The calibration curve is surfaced live in the frontend under Live 2026 → Predictive.

---

## Quickstart

```bash
git clone https://github.com/ritvikv03/MM.git
cd MM

# Install everything (Python + Node)
make setup

# Fill in .env: KAGGLE_USERNAME, KAGGLE_KEY, WANDB_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY
```

```bash
# Start backend + frontend together
make dev

# Or individually
make dev-api          # FastAPI on :8000
make dev-frontend     # Next.js on :3000
```

```bash
# Pipeline runs
make pipeline         # Full 6AM run
make pipeline-intel   # 12PM intel update
make pipeline-results # 10PM results ingestion

# Via CLI
mm run full --season 2026
mm run intel --dry-run    # dry-run skips all network/DB calls
```

```bash
# Tests
make test             # pytest + vitest
make test-coverage    # pytest with 80% coverage threshold
```

---

## Project Structure

```
MM/
├── Makefile                     # make setup / dev / pipeline / test
├── src/
│   ├── cli.py                   # mm CLI — mm run [full|intel|results]
│   ├── api/                     # FastAPI server + engines
│   │   ├── intel_engine.py      # Autonomous intel flag generation
│   │   ├── matchup_engine.py
│   │   ├── bracket_runner.py
│   │   └── bracket_2026.py
│   ├── data/                    # One ingestion module per source
│   ├── graph/                   # PyG heterogeneous graph construction
│   ├── model/
│   │   ├── gat_encoder.py       # Graph Attention Network
│   │   ├── bayesian_head.py     # PyMC posterior inference
│   │   ├── calibration.py       # Isotonic regression calibrator
│   │   └── skellam.py           # Zero-truncated Skellam distribution
│   ├── simulation/              # Monte Carlo + Chaos Engine + Copula
│   ├── betting/                 # Kelly, CLV, Black-Scholes, Prospect Theory
│   └── pipeline/                # PipelineRunner (full/intel/results)
├── frontend/                    # Next.js 15 — deployed on Vercel
│   ├── components/
│   │   ├── projections/         # Live 2026 (intel feed, predictions)
│   │   ├── rankings/            # Power rankings
│   │   ├── matchup/             # Matchup Oracle
│   │   ├── bracket/             # Bracket Architect + heatmap
│   │   └── warroom/             # Leverage War Room
│   └── lib/
│       ├── api.ts               # API client + TypeScript interfaces
│       ├── team-data.ts         # 2026 tournament team metadata
│       └── hooks/use-live-data.ts  # SWR polling hooks (5-min refresh)
├── tests/                       # pytest suite mirroring src/
├── .github/workflows/
│   ├── daily_pipeline.yml       # 3× daily cron (6AM/12PM/10PM ET)
│   └── pr_tests.yml             # CI on every PR
└── supabase/migrations/         # Postgres schema + indexes
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Graph Neural Network | PyTorch Geometric — GAT + heterogeneous graph |
| Bayesian Inference | PyMC 5 + ArviZ (ADVI / NUTS) |
| API | FastAPI + Uvicorn |
| Data | pandas, polars, numpy, scipy |
| Scraping | requests, httpx, BeautifulSoup, Playwright |
| MLOps | Weights & Biases |
| Frontend | Next.js 15 (App Router), Tailwind CSS, shadcn/ui |
| Charts | Recharts + D3 KDE |
| Animation | Framer Motion |
| Data Fetching | SWR (5-min auto-poll) |
| Validation | Zod |
| Testing | pytest + vitest + Playwright |
| Database | Supabase (Postgres + Realtime) |
| Deployment | Vercel (frontend) + GitHub Actions (pipeline) |

---

## License

MIT — use freely, cite honestly.
