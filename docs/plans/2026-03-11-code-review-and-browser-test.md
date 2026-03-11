# Code Review, Simplify, and Browser Validation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run code-review + simplify on every frontend module, fix all issues found, update CLAUDE.md with session learnings, then verify the full UI in a browser with Playwright.

**Architecture:** Module-by-module: review → simplify → fix → vitest green → move to next module. Final step is Playwright end-to-end browser validation of all 6 nav pages.

**Tech Stack:** Next.js 14, TypeScript, React Three Fiber, Framer Motion, Vitest, Playwright, PyMC/PyTorch backend (pytest)

---

## Module Order
1. `BracketSimulator.tsx` + `BracketHeatmap.tsx` (most complex, 840 LOC)
2. `ConstellationCanvas.tsx` + `BasketballCourt3D.tsx` + `TeamNode.tsx` (3D layer)
3. `api.ts` + `api-types.ts` + `bracket-utils.ts` (data layer)
4. CLAUDE.md revision via claude-md-management skill
5. Playwright browser validation (all 6 pages + bracket cascade)

---

### Task 1: Code-Review BracketSimulator.tsx

**Files:**
- Review: `frontend/components/bracket/BracketSimulator.tsx`
- Review: `frontend/components/bracket/BracketHeatmap.tsx`

**Step 1: Invoke code-review skill on BracketSimulator**

Run skill: `code-review:code-review`
Focus on: computeBracket purity, cascade correctness, missing null guards, React key stability, performance (useMemo deps).

**Step 2: Invoke simplify skill**

Run skill: `simplify`
Look for: duplicate logic between MiniGameCell/LargeGameCell, unnecessary re-renders, dead code.

**Step 3: Apply all critical fixes**

Priority fixes to apply (examples — actual list from review):
- Add `position: 'relative'` to MiniGameCell wrapper (needed for absolute USER badge)
- Guard `estimateSpread` against null teamA/teamB
- Stabilize `useMemo` deps array for `games` (avoid object recreation)
- Remove any `any` types

**Step 4: Run vitest**

```bash
cd frontend && npx vitest run
```
Expected: 70 tests pass

**Step 5: Commit**

```bash
git add frontend/components/bracket/
git commit -m "refactor(bracket): code-review fixes — null guards, position:relative, stable deps"
```

---

### Task 2: Code-Review ConstellationCanvas + 3D Components

**Files:**
- Review: `frontend/components/constellation/ConstellationCanvas.tsx`
- Review: `frontend/components/constellation/BasketballCourt3D.tsx`
- Review: `frontend/components/constellation/TeamNode.tsx`
- Review: `frontend/components/constellation/ConferenceNode.tsx`
- Review: `frontend/components/constellation/GameEdge.tsx`

**Step 1: Invoke code-review skill**

Run skill: `code-review:code-review`
Focus on: SSR guards (`typeof window !== 'undefined'`), Three.js memory leaks (dispose textures), React Three Fiber hook usage, camera position correctness.

**Step 2: Invoke simplify skill**

Run skill: `simplify`
Look for: duplicate lighting setup, unused imports, redundant useMemo.

**Step 3: Apply fixes**

Key expected fixes:
- Ensure `BasketballCourt3D` CanvasTexture is disposed on unmount via `useEffect` cleanup
- Verify `hemisphereLight` args type is correct for R3F
- Add `frustumCulled={false}` to large geometry if needed

**Step 4: Run vitest**

```bash
cd frontend && npx vitest run __tests__/constellation.test.ts
```
Expected: 8 tests pass

**Step 5: Commit**

```bash
git add frontend/components/constellation/
git commit -m "refactor(constellation): dispose CanvasTexture, fix hemisphere light typing"
```

---

### Task 3: Code-Review API + Data Layer

**Files:**
- Review: `frontend/lib/api.ts`
- Review: `frontend/lib/api-types.ts`
- Review: `frontend/lib/bracket-utils.ts`
- Review: `frontend/lib/mock-data.ts`

**Step 1: Invoke code-review skill**

Run skill: `code-review:code-review`
Focus on: Zod schema completeness, error surfacing (no swallowed catch), type safety of EnrichedMatchupResponse merge, mock-data generator accuracy.

**Step 2: Invoke simplify skill**

Run skill: `simplify`
Look for: repeated fetch patterns, over-engineered mock generators, unused exports.

**Step 3: Apply fixes**

Key expected fixes:
- Add explicit error logging in catch blocks (not silent fallback)
- Validate merged matchup response against MatchupResponseSchema before returning
- Remove unused exports from mock-data.ts

**Step 4: Run vitest**

```bash
cd frontend && npx vitest run __tests__/api-types.test.ts __tests__/matchup.test.ts
```
Expected: 11 tests pass

**Step 5: Commit**

```bash
git add frontend/lib/
git commit -m "refactor(api): surface errors, validate merged response, remove dead exports"
```

---

### Task 4: CLAUDE.md Revision

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Invoke claude-md-management skill**

Run skill: `claude-md-management:revise-claude-md`
Add session learnings:
- BracketSimulator computeBracket pattern (userPicks Map → pure derivation)
- THREE.Color hex-string fix
- BasketballCourt3D SSR pattern with `typeof window` guard
- bracket-utils: rgb() format requirement for probToHeatColor

**Step 2: Verify CLAUDE.md is under 400 lines (truncation safety)**

```bash
wc -l CLAUDE.md
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude-md): session learnings — bracket cascade, THREE.Color fix, court SSR"
```

---

### Task 5: Playwright Browser Validation

**Files:**
- Create: `frontend/e2e/bracket.spec.ts`
- Create: `frontend/playwright.config.ts`

**Step 1: Install Playwright**

```bash
cd frontend && npx playwright install --with-deps chromium
```

**Step 2: Write playwright.config.ts**

```typescript
import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './e2e',
  use: { baseURL: 'http://localhost:3000', headless: true },
  webServer: { command: 'npm run dev', url: 'http://localhost:3000', reuseExistingServer: true },
});
```

**Step 3: Write bracket cascade e2e test**

```typescript
// e2e/bracket.spec.ts
import { test, expect } from '@playwright/test';

test('bracket page loads and shows R64 games', async ({ page }) => {
  await page.goto('/');
  await page.click('[title="Bracket"]');
  await expect(page.locator('text=BRACKET ENGINE')).toBeVisible();
  await expect(page.locator('text=EAST REGION')).toBeVisible();
  await expect(page.locator('text=R64')).toBeVisible();
});

test('clicking team picks winner and cascades', async ({ page }) => {
  await page.goto('/');
  await page.click('[title="Bracket"]');
  await page.waitForSelector('text=EAST REGION');
  // Champion banner should show a team name
  await expect(page.locator('text=PREDICTED CHAMPION')).toBeVisible();
});

test('graph page shows constellation', async ({ page }) => {
  await page.goto('/');
  await page.click('[title="Graph"]');
  await expect(page.locator('canvas')).toBeVisible();
});

test('all nav pages render without crash', async ({ page }) => {
  await page.goto('/');
  for (const label of ['Rankings', 'Matchup', '2026', 'War Room']) {
    await page.click(`[title="${label}"]`);
    await expect(page).not.toHaveTitle('Error');
  }
});
```

**Step 4: Start dev server and run Playwright**

```bash
# Terminal 1 (background):
cd frontend && npm run dev &

# Terminal 2:
cd frontend && npx playwright test --reporter=list
```

Expected: 4 tests pass, no page crashes

**Step 5: Commit**

```bash
git add frontend/e2e/ frontend/playwright.config.ts
git commit -m "test(e2e): playwright suite — bracket cascade, graph page, all nav pages"
```

---

### Task 6: Final Verification

**Step 1: Run full test suite**

```bash
# Backend
python -m pytest tests/ -q

# Frontend unit
cd frontend && npx vitest run

# Frontend e2e
cd frontend && npx playwright test
```

Expected: 1018 backend + 70 unit + 4 e2e = all pass

**Step 2: TypeScript clean check**

```bash
cd frontend && npx tsc --noEmit
```
Expected: 0 errors

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final verification — all tests pass, e2e green, 0 TS errors"
```
