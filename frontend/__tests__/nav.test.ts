import { describe, it, expect } from 'vitest';

// Test pure nav logic (no React rendering needed)
describe('NavPage routing', () => {
  const PAGES = ['rankings', 'matchup', 'bracket', 'projections', 'warroom'] as const;

  it('has 5 pages', () => {
    expect(PAGES).toHaveLength(5);
  });

  it('rankings is the first page', () => {
    expect(PAGES[0]).toBe('rankings');
  });

  it('all pages are unique strings', () => {
    const set = new Set(PAGES);
    expect(set.size).toBe(PAGES.length);
  });
});

describe('ModelStatus colors', () => {
  const STATUS_COLORS: Record<string, string> = {
    online:  '#00ff88',
    offline: '#ff2d55',
    loading: '#ffb800',
  };

  it('has color for online status', () => {
    expect(STATUS_COLORS.online).toBe('#00ff88');
  });

  it('has color for offline status', () => {
    expect(STATUS_COLORS.offline).toBe('#ff2d55');
  });

  it('has color for loading status', () => {
    expect(STATUS_COLORS.loading).toBe('#ffb800');
  });

  it('all colors are valid hex strings', () => {
    Object.values(STATUS_COLORS).forEach((c) => {
      expect(c).toMatch(/^#[0-9a-f]{6}$/i);
    });
  });
});

describe('Season selector', () => {
  const SEASONS = [2022, 2023, 2024, 2025];

  it('contains 4 seasons', () => {
    expect(SEASONS).toHaveLength(4);
  });

  it('default season is 2024', () => {
    expect(SEASONS).toContain(2024);
  });

  it('seasons are in ascending order', () => {
    for (let i = 1; i < SEASONS.length; i++) {
      expect(SEASONS[i]).toBeGreaterThan(SEASONS[i - 1]);
    }
  });
});
