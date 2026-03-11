# Session Tasks — Frontend Overhaul + Interactive All-Rounds Bracket
_Completed: 2026-03-11_

## Completed
- [x] Step 1 — CLAUDE.md §10 agentic workflow rules (8 rules appended)
- [x] Step 2 — tasks/ directory bootstrap (todo.md + lessons.md)
- [x] Step 3 — Fix api.ts: fetchMatchup now merges real API p_win_home/spread_mean/samples; mock enrichment fills remaining fields
- [x] Step 4 — Fix page.tsx: useEffect([season]) triggers fetchGraph → setGraphData; 'graph' nav page renders ConstellationCanvas
- [x] Step 5 — Fix TeamNode.tsx: THREE.Color(`#${node.color.toString(16).padStart(6,'0')}`) — hex string not raw int
- [x] Step 6 — BasketballCourt3D.tsx: canvas-texture NCAA court on PlaneGeometry(22,11.7), hardwood gradient + all court lines + backboards
- [x] Step 7 — ConstellationCanvas.tsx: camera [0,14,10] overhead, HemisphereLight, removed Stars, added BasketballCourt3D
- [x] Step 8 — Sidebar.tsx: added 'graph' to NavPage type + GraphIcon + NAV_ITEMS entry
- [x] Step 9 — BracketSimulator.tsx: full 6-round rewrite with computeBracket() pure function, BracketGame state machine, cascade logic, user picks Map, RegionPanel + MiniGameCell + LargeGameCell, Interrogator sidebar, chaos slider, WPA sliders, reset button, simulate button
- [x] Step 10 — BracketHeatmap.tsx: SVG grid, teams sorted by Championship prob, probToHeatColor cells, round labels
- [x] Step 11 — bracket-utils.ts: added getAdvancementProb, rankByRound, normalizeEntropy, topChampionshipContenders; fixed probToHeatColor to return rgb() format with clamping

## Verification Results
- [x] `python -m pytest tests/ -q` — 1018 passed, 7 skipped
- [x] `npx vitest run` — 70 passed (all 8 test files)
- [x] `npx tsc --noEmit` — 0 errors
- [ ] Live: Bracket R64 → Championship cascade (requires browser)
- [ ] Live: Graph page basketball court (requires browser)
- [ ] Live: Season change reloads graph (requires browser)
