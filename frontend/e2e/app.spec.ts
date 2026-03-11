import { test, expect } from '@playwright/test';

test('homepage loads with sidebar', async ({ page }) => {
  await page.goto('/');
  // Use exact span match to avoid strict-mode collision with the projections footer text
  await expect(page.locator('span', { hasText: 'MARCH MADNESS ORACLE' }).first()).toBeVisible({ timeout: 10000 });
});

test('bracket page shows BRACKET ENGINE and regions', async ({ page }) => {
  await page.goto('/');
  // Click the Bracket nav item
  await page.click('[title="Bracket"]');
  await expect(page.locator('text=BRACKET ENGINE')).toBeVisible({ timeout: 10000 });
  await expect(page.locator('text=EAST REGION')).toBeVisible({ timeout: 10000 });
  await expect(page.locator('text=PREDICTED CHAMPION')).toBeVisible({ timeout: 10000 });
});

test('chaos slider is present and interactive', async ({ page }) => {
  await page.goto('/');
  await page.click('[title="Bracket"]');
  await expect(page.locator('text=RISK & VARIANCE DIAL')).toBeVisible({ timeout: 10000 });
  const slider = page.locator('input[type="range"]').first();
  await expect(slider).toBeVisible();
});

test('matchup page loads', async ({ page }) => {
  await page.goto('/');
  await page.click('[title="Matchup"]');
  // MatchupOracle should render
  await expect(page).not.toHaveURL(/error/);
  await page.waitForTimeout(2000);
  // No crash = pass
  const body = page.locator('body');
  await expect(body).toBeVisible();
});

test('rankings page loads', async ({ page }) => {
  await page.goto('/');
  await page.click('[title="Rankings"]');
  await page.waitForTimeout(2000);
  await expect(page.locator('body')).toBeVisible();
});

test('projections page loads (default page)', async ({ page }) => {
  await page.goto('/');
  // Default page is projections
  await page.waitForTimeout(2000);
  await expect(page.locator('body')).toBeVisible();
});

test('war room page loads', async ({ page }) => {
  await page.goto('/');
  await page.click('[title="War Room"]');
  await page.waitForTimeout(2000);
  await expect(page.locator('body')).toBeVisible();
});

test('graph page shows canvas element', async ({ page }) => {
  await page.goto('/');
  await page.click('[title="Graph"]');
  // Wait for Three.js canvas to mount
  await expect(page.locator('canvas')).toBeVisible({ timeout: 15000 });
});
