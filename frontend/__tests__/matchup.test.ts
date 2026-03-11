import { describe, it, expect } from 'vitest';
import { silvermanBandwidth, linspace, computeKDE, kdeIntersectionArea } from '../lib/d3-kde';

describe('RidgelinePlot KDE math', () => {
  it('bandwidth is positive for real win probability samples', () => {
    const samples = Array.from({ length: 100 }, (_, i) => 0.4 + i * 0.002);
    expect(silvermanBandwidth(samples)).toBeGreaterThan(0);
  });

  it('KDE integrates to approximately 1 over [0,1]', () => {
    const samples = [0.3, 0.4, 0.5, 0.6, 0.7, 0.55, 0.45, 0.5];
    const grid = linspace(0, 1, 200);
    const density = computeKDE(samples, grid);
    // Trapezoidal integration
    let area = 0;
    for (let i = 0; i < grid.length - 1; i++) {
      area += (density[i] + density[i + 1]) / 2 * (grid[i + 1] - grid[i]);
    }
    expect(area).toBeCloseTo(1.0, 1);
  });

  it('intersection area between identical distributions equals full area', () => {
    const grid = linspace(0, 1, 100);
    const d = grid.map(() => 1.0);
    const area = kdeIntersectionArea(d, d, grid);
    expect(area).toBeCloseTo(1.0, 1);
  });

  it('intersection area between non-overlapping is near zero', () => {
    const grid = linspace(0, 1, 100);
    // d1 concentrated at left, d2 at right
    const d1 = grid.map((x) => x < 0.2 ? 5.0 : 0.0);
    const d2 = grid.map((x) => x > 0.8 ? 5.0 : 0.0);
    const area = kdeIntersectionArea(d1, d2, grid);
    expect(area).toBeLessThan(0.01);
  });

  it('linspace generates monotone sequence', () => {
    const pts = linspace(0, 1, 50);
    for (let i = 1; i < pts.length; i++) {
      expect(pts[i]).toBeGreaterThan(pts[i - 1]);
    }
  });

  it('computeKDE returns non-negative densities', () => {
    const samples = [0.3, 0.5, 0.7];
    const grid = linspace(0, 1, 50);
    const density = computeKDE(samples, grid);
    density.forEach((d) => expect(d).toBeGreaterThanOrEqual(0));
  });

  it('KDE peak is near the median of samples', () => {
    // Use a tight cluster around 0.5 with small spread so bandwidth > 0
    const samples = [0.48, 0.49, 0.50, 0.51, 0.52, 0.50, 0.50];
    const grid = linspace(0, 1, 200);
    const density = computeKDE(samples, grid);
    const peakIdx = density.indexOf(Math.max(...density));
    expect(grid[peakIdx]).toBeCloseTo(0.5, 1);
  });
});

describe('Spread samples statistics', () => {
  it('mean of centered normal is near zero', () => {
    const samples = Array.from({ length: 1000 }, (_, i) => (i - 500) * 0.02);
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(mean).toBeCloseTo(0, 1);
  });
});
