# Madness Matrix — Full System Overhaul Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans then superpowers:subagent-driven-development to implement this design.

**Approved:** 2026-03-12

---

## 1. Product Vision

**Madness Matrix** is a deployed, production-grade NCAA March Madness prediction engine that:
- Runs autonomously 3x daily via GitHub Actions (no human trigger needed)
- Predicts the 2026 bracket using the highest-level ML/stat/CS methodology available
- Stores every prediction snapshot in Supabase so model accuracy is historically trackable
- Surfaces descriptive, predictive, and prescriptive analytics in a modern deployed frontend
- Never uses synthetic or stub data — real data only, always

---

## 2. Infrastructure (Zero Cost)

```
GitHub Actions (public repo → unlimited free minutes)
  Cron: 6 AM ET / 12 PM ET / 10 PM ET daily
  → Scrape all sources → run full ML stack → write to Supabase

Vercel (Hobby plan — free)
  → Next.js 15 frontend deployed at production URL
  → Vercel API routes for lightweight on-demand calculations
  → Reads Supabase directly (no separate backend server)

Supabase (free project — 500MB Postgres + Realtime)
  → Stores teams, predictions, bracket_runs, alerts, snapshots
  → Realtime: intel_alerts table pushes to frontend live

GitHub repo: github.com/ritvikv03/MM (public, already active)
```

### Environment Variables Required
```
SUPABASE_URL              # Supabase project URL
SUPABASE_SERVICE_KEY      # Supabase service role key (server-side only)
SUPABASE_ANON_KEY         # Supabase anon key (frontend-safe)
KAGGLE_USERNAME           # Kaggle API credentials
KAGGLE_KEY
WANDB_API_KEY             # W&B experiment tracking
WANDB_PROJECT
WANDB_ENTITY
VERCEL_CRON_SECRET        # Shared secret to authenticate Vercel → Actions webhook
```

---

## 3. GitHub Actions — 3x Daily Pipeline

```yaml
# .github/workflows/daily_pipeline.yml
on:
  schedule:
    - cron: '0 11 * * *'   # 6 AM ET  — full run: scrape + train + predict
    - cron: '0 17 * * *'   # 12 PM ET — intel refresh + partial re-score
    - cron: '0 3 * * *'    # 10 PM ET — ingest game results + Brier update
  workflow_dispatch:        # manual trigger always available
```

### Run Profiles

| Run | Label | Scrapes | Computes | Supabase Writes |
|-----|-------|---------|----------|-----------------|
| 6 AM | `full` | All sources (Barttorvik, ESPN, Rotowire, On3, NBA.com, Sports Ref, Kaggle) | Full ST-GNN → Bayesian ADVI → Monte Carlo 10k → Calibration → CLV → RL | `teams`, `matchup_predictions`, `bracket_runs`, `prediction_snapshots`, `intel_alerts`, `pipeline_runs` |
| 12 PM | `intel` | Intel only (Rotowire, ESPN injuries, Reddit, On3) | Re-score alerts, cross-source severity bump, temporal Bayesian warm-start update | `intel_alerts`, `prediction_snapshots` (delta only) |
| 10 PM | `results` | Kaggle game results (tournament only) | Brier Score + Log-Loss vs. actuals, isotonic recalibration | `game_results`, `prediction_snapshots.brier_score`, `pipeline_runs` |

---

## 4. Database Schema (Supabase)

```sql
teams (id, season, name, conference, seed, adj_oe, adj_de, adj_em, tempo,
       luck, sos, coach, region, scraped_at)

matchup_predictions (id, season, team_a, team_b, p_win_a, spread_mean,
                     spread_std, samples_json, model_version, computed_at)

bracket_runs (id, season, run_date, n_simulations, advancement_probs,
              champion_prob, model_version, runtime_secs, computed_at)

prediction_snapshots (id, season, snapshot_date, team, championship_prob,
                      final_four_prob, elite_eight_prob, sweet_sixteen_prob,
                      brier_score, log_loss, created_at)

intel_alerts (id, alert_id, source, content, keywords, severity,
              teams_mentioned, region, needs_verification, url,
              alerted_at, resolved, created_at)

game_results (id, season, game_date, home_team, away_team, home_score,
              away_score, neutral_site, tournament_round, created_at)

pipeline_runs (id, run_date, trigger, status, teams_updated,
               predictions_computed, alerts_found, duration_secs,
               error_log, created_at)
```

Supabase Realtime enabled on: `intel_alerts`, `prediction_snapshots`

---

## 5. Prediction Engine Stack

Full pipeline executes in sequence during the 6 AM run:

```
1. Data Ingestion
   Barttorvik T-Rank (AdjO, AdjD, Tempo, Luck, PORPAGATU!, BPM rolling EWMA)
   + Sports Reference CBB (Four Factors, shot types, coaching ATS records)
   + Kaggle March Mania (historical results, seeds, team spellings)
   + Rotowire (injuries) + ESPN CBB (injury cross-check)
   + NBA.com Early Entry (NBA Draft departures — official)
   + On3 Transfer Portal (transfer moves)

2. Graph Construction (src/graph/)
   Heterogeneous PyG graph:
   - Team nodes: 12-dim features (efficiency + entropy + availability + continuity)
   - Conference nodes: conference-level RPI aggregates (NEW)
   - Game edges: margin/efficiency delta
   - Edge features: court location, rest disparity, travel fatigue (distance + TZ + altitude)
   Shannon Entropy injected as node feature per team

3. ST-GNN Inference (src/model/)
   GAT encoder: 4-head attention, 2 layers, 64-dim hidden
   LSTM temporal: 2-layer, 128-dim → rolling game momentum
   Output: 128-dim team embeddings

4. Bayesian Head (src/model/bayesian_head.py)
   PyMC ADVI, 5,000 iterations
   - Coach ATS hierarchical prior (Tom Izzo Effect)
   - Clutch/luck regression shrinkage (pm.Normal(0.5, sigma≤0.15))
   - Zero-Truncated Skellam spread likelihood
   Output: 2,000 posterior samples per matchup

5. NEW: Isotonic Regression Calibration (src/model/calibration.py)
   Fits monotone calibration curve on 2012–2024 Brier scores
   Corrects overconfident chalk predictions
   Output: calibrated_p_win per matchup stored alongside raw p_win

6. Ensemble Voting (src/model/ensemble.py)
   Fundamentalist (adj_oe/adj_de/rebounding)
   Market Reader (historical spread movement + sharp money)
   Chaos Agent (travel fatigue, altitude, momentum, officiating bias)
   Output: consensus + dissent reasons stored in matchup_predictions

7. NEW: Temporal Bayesian Warm-Start (src/model/bayesian_head.py)
   12 PM and 10 PM runs use morning posterior as prior
   Updates only on delta from new game results
   10x faster, statistically correct sequential updating

8. Monte Carlo Bracket Simulation (src/simulation/monte_carlo.py)
   10,000 full bracket trials (up from 1,000)
   Chaos Engine: 1/2-seed elimination → region topology reweighting
   Gaussian Copula: conference contagion on upsets
   RL Bracket Optimizer: 3 variants (Chalk / Leverage / Chaos)
   Output: advancement_probs per team per round

9. NEW: Conference RPI Weighting (src/graph/node_features.py)
   Conference strength hierarchy from Sports Reference
   Injected into Conference node features for the GAT
   Enables better cross-conference strength modeling

10. CLV + Betting Layer
    Prospect Theory CLV scanner (Prelec weighting, 5v12/4v13 peak)
    Black-Scholes Vega hedge recommendations per round
    Fractional Kelly sizing per matchup
    All written to bracket_runs + prediction_snapshots
```

---

## 6. Frontend Architecture

### Branding
- **Name:** Madness Matrix
- **Tagline:** "Quantitative bracket forecasting powered by ST-GNN + Bayesian inference"
- **Color palette:** Deep navy `#0a0f1e` base, electric blue `#3b82f6` primary, amber `#f59e0b` accent, red `#ef4444` alert
- **Typography:** Inter (body), JetBrains Mono (numbers/metrics)

### Tech Stack
```
Next.js 15 (App Router)
Tailwind CSS + shadcn/ui
@tanstack/react-query (server state)
@supabase/supabase-js (database + Realtime)
Recharts (bracket heatmap, sparklines, calibration curve)
Framer Motion (page transitions)
Zod (schema validation)
date-fns (date formatting)

REMOVED: Three.js, @react-three/fiber, @react-three/drei
REMOVED: lib/mock-data.ts
REMOVED: USE_REAL_DATA env var and all stub code
REMOVED: StubDataBanner
```

### Navigation (5 tabs, season-aware global state)

```
[🔴 Live 2026] [📊 Rankings] [⚔ Matchup] [🏆 Bracket] [🎯 War Room]

Season: [2026 ▾]  ← global selector, all tabs update simultaneously
```

Switching season:
- 2026 (current): Live tab enabled, all tabs show real-time predictions
- 2012–2025 (historical): Live tab shows historical descriptive view,
  actual results overlaid on predictions as green ✓ / red ✗

### Tab 1 — Live 2026

Sub-views: `[Descriptive] [Predictive] [Prescriptive]`

**Descriptive:**
- Current efficiency leaderboard (adj_em trending sparkline)
- Conference RPI tier visualization
- Games played results feed
- Injury/alert timeline from intel_alerts (Supabase Realtime)

**Predictive:**
- Live bracket probabilities: Championship%, F4%, E8% per team
- Probability drift chart: how odds shifted across 3 daily runs
  (temporal Bayesian updating made visible)
- Model calibration curve: isotonic-corrected vs. raw posterior
- Model confidence interval bands on all predictions

**Prescriptive:**
- Top 10 bracket picks ranked by CLV magnitude (Prospect Theory)
- Active Intel alerts feed (Realtime, with UNVERIFIED badge for Reddit)
- 3 RL bracket variant cards: Chalk / Leverage / Chaos
- Kelly-sized confidence indicators per team

### Tab 2 — Rankings

- T-Rank table: adj_oe, adj_de, adj_em, tempo, luck, sos
- Conference RPI tier badge per team (★★★★★ → ★☆☆☆☆)
- Calibrated championship probability column
- 14-day adj_em trend sparkline per team
- Region color coding (East/West/South/Midwest)
- Historical: frozen pre-tournament snapshot + actual results column

### Tab 3 — Matchup Oracle

- Team selector (search from Supabase teams table)
- Ridgeline KDE plot of pre-computed 2,000 posterior spread samples
- Win probability with credible intervals (10th/50th/90th percentile)
- Ensemble dissent panel: Fundamentalist / Market Reader / Chaos Agent votes
- Temporal drift badge: if spread shifted >1.5 pts across daily runs
- Intel alert badge: if either team has active unresolved alerts
- Historical: actual result overlaid on posterior distribution

### Tab 4 — Bracket

- Full 6-round interactive bracket (click to pick winners)
- Monte Carlo heatmap: advancement probabilities from Supabase bracket_runs
- Chaos Engine indicator: eliminated titan → affected region lights up amber
- Calibrated vs. uncalibrated probability toggle
- Historical: bracket locked to actual results, model ghost overlay for review

### Tab 5 — War Room

- Leverage matrix heatmap (CLV × public ownership)
- Black-Scholes Vega panel per round
- Prospect Theory CLV scanner: ranked value picks
- 3 bracket variant cards
- Bracket portfolio strategy guide
- Kelly sizing calculator

---

## 7. Files to Delete (No Synthetic Data Mandate)

```
frontend/lib/mock-data.ts         → DELETE
src/api/server.py _build_stub_*   → DELETE all stub functions
src/api/server.py _TEAMS_BY_CONF  → DELETE (move to Supabase)
```

All API routes read from Supabase. No fallback stub. If Supabase is
unreachable, return HTTP 503 with clear error — never fabricate data.

---

## 8. New Files to Create

```
Backend (GitHub Actions pipeline):
  src/pipeline/supabase_writer.py    — write all predictions to Supabase
  src/pipeline/github_actions_runner.py — orchestrates full/intel/results runs
  src/model/calibration.py           — isotonic regression calibration
  src/data/conference_rpi.py         — conference RPI computation
  .github/workflows/daily_pipeline.yml
  .github/workflows/pr_tests.yml

Frontend:
  frontend/lib/supabase.ts           — Supabase client singleton
  frontend/lib/queries.ts            — React Query hooks for all data
  frontend/components/live/LiveTab.tsx
  frontend/components/live/DescriptiveView.tsx
  frontend/components/live/PredictiveView.tsx
  frontend/components/live/PrescriptiveView.tsx
  frontend/components/live/CalibrationCurve.tsx
  frontend/components/live/ProbabilityDrift.tsx
  frontend/components/intel/IntelFeed.tsx
  frontend/components/rankings/TrendSparkline.tsx
  frontend/components/matchup/EnsembleDissent.tsx

Config:
  supabase/migrations/001_initial_schema.sql
  vercel.json (cron fallback trigger)
```

---

## 9. CLAUDE.md Updates Required

- §1: Update primary goal to include "deployed production system"
- §2: Add Supabase, Vercel, GitHub Actions to tech stack
- New §15: Deployment Architecture
- New §16: No Synthetic Data — Hard Rule
- New §17: Frontend Stack Standards (Tailwind/shadcn/Recharts)
- Update branding references: "Ethereal Oracle" → "Madness Matrix" throughout
