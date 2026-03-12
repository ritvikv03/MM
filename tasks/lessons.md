# Lessons Learned
_Append-only log. Each entry: date + lesson._

## 2026-03-11
- `THREE.Color(node.color)` fails when `node.color` is a raw integer (hex number like `0xff6b35`). Must convert: `new THREE.Color(`#${node.color.toString(16).padStart(6, '0')}`)`.
- `fetchMatchup` in api.ts was ignoring real API responses and always falling back to mock. Always parse `res.json()` before constructing the enriched response.
- ConstellationCanvas was never rendered in page.tsx — the 'graph' nav page didn't exist. Required adding NavPage type + Sidebar item + dynamic import.
- BracketSimulator only showed R64 matchups. The `bracketResults` computed all rounds but the JSX only rendered `data.rounds[0]`. Needed a full BracketGame state machine with cascade logic.
- The bracket `computeBracket()` pure function pattern (userPicks Map → derived games array) is cleaner than storing winners in React state directly — chaos/WPA changes auto-recompute all non-user-picked games.
- For the Three.js basketball court, use `document.createElement('canvas')` + `CanvasTexture` (not an image import) so it works in SSR/RSC environments with a `typeof window !== 'undefined'` guard.

## 2026-03-11 (session 2)
- Vitest will pick up Playwright e2e spec files if `exclude` is not set — always add `exclude: ['e2e/**', 'node_modules/**']` to vitest.config.ts in projects that colocate e2e and unit tests.
- Files created during a development session can be untracked in git even if tests run locally. Always audit `git ls-files` vs `find . -name "*.py"` before declaring a phase complete.
- `requirements.txt` must include `fastapi` + `uvicorn` when the project ships a FastAPI server — the API cannot start without them, even if all tests pass.
- Hardcoded relative paths (e.g. `open("data/foo.json", "w")`) fail when pytest or any runner changes CWD. Always use `pathlib.Path(__file__).parent / ...` for paths relative to the source file.
