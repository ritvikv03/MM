# Lessons Learned
_Append-only log. Each entry: date + lesson._

## 2026-03-11
- `THREE.Color(node.color)` fails when `node.color` is a raw integer (hex number like `0xff6b35`). Must convert: `new THREE.Color(`#${node.color.toString(16).padStart(6, '0')}`)`.
- `fetchMatchup` in api.ts was ignoring real API responses and always falling back to mock. Always parse `res.json()` before constructing the enriched response.
- ConstellationCanvas was never rendered in page.tsx — the 'graph' nav page didn't exist. Required adding NavPage type + Sidebar item + dynamic import.
- BracketSimulator only showed R64 matchups. The `bracketResults` computed all rounds but the JSX only rendered `data.rounds[0]`. Needed a full BracketGame state machine with cascade logic.
- The bracket `computeBracket()` pure function pattern (userPicks Map → derived games array) is cleaner than storing winners in React state directly — chaos/WPA changes auto-recompute all non-user-picked games.
- For the Three.js basketball court, use `document.createElement('canvas')` + `CanvasTexture` (not an image import) so it works in SSR/RSC environments with a `typeof window !== 'undefined'` guard.
