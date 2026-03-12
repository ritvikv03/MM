# Session Tasks — 2026-03-11 Repository Audit & Hardening

## Completed
- [x] Step 1 — Fixed vitest.config.ts: added `exclude: ['e2e/**', 'node_modules/**']` — Vitest was picking up Playwright e2e spec and crashing due to duplicate `@playwright/test` version
- [x] Step 2 — Audited all 19 untracked src/ files and 10 test files before staging; no logic errors or stubs returning None found
- [x] Step 3 — Staged and committed all untracked backend modules (33 files, 4665 lines): API server, Phase 5–9 betting/simulation/data/model modules, all corresponding test files, pyproject.toml, requirements.txt, .gitignore
- [x] Step 4 — Fixed requirements.txt: added `fastapi>=0.115` + `uvicorn[standard]>=0.30` (API server was locally unrunnable without these)
- [x] Step 5 — Fixed news_scraper.py:105: replaced hardcoded `"data/asymmetry_alerts.json"` with `pathlib.Path(__file__).parent.parent.parent / "data"` — would fail when CWD ≠ project root

## Verification Results
- [x] `python -m pytest tests/ -q` → 1018 passed, 7 skipped (torch_geometric not installed — intentional)
- [x] `cd frontend && npx vitest run` → 8 files, 70 tests, 0 failures
- [x] `npx tsc --noEmit` → 0 errors
- [x] `git push origin main` → pushed commit `0537d35`
