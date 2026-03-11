import { describe, it, expect } from 'vitest';
import {
  epanechnikovKernel,
  silvermanBandwidth,
  linspace,
  computeKDE,
  kdeIntersectionArea,
} from '../lib/d3-kde';

describe('epanechnikovKernel', () => {
  it('returns 0 outside bandwidth', () => {
    const k = epanechnikovKernel(1);
    expect(k(5, 0)).toBe(0);
  });
  it('returns max at center', () => {
    const k = epanechnikovKernel(1);
    expect(k(0, 0)).toBeCloseTo(0.75);
  });
});

describe('silvermanBandwidth', () => {
  it('returns positive for non-empty data', () => {
    const bw = silvermanBandwidth([0.1, 0.3, 0.5, 0.7, 0.9]);
    expect(bw).toBeGreaterThan(0);
  });
  it('handles single element', () => {
    expect(silvermanBandwidth([0.5])).toBe(1);
  });
});

describe('linspace', () => {
  it('generates correct number of points', () => {
    expect(linspace(0, 1, 5)).toHaveLength(5);
  });
  it('starts and ends correctly', () => {
    const pts = linspace(0, 1, 3);
    expect(pts[0]).toBeCloseTo(0);
    expect(pts[2]).toBeCloseTo(1);
  });
  it('returns empty for n=0', () => {
    expect(linspace(0, 1, 0)).toHaveLength(0);
  });
});

describe('computeKDE', () => {
  it('returns array same length as grid', () => {
    const grid = linspace(0, 1, 20);
    const density = computeKDE([0.3, 0.5, 0.7], grid);
    expect(density).toHaveLength(20);
  });
  it('density values are non-negative', () => {
    const grid = linspace(0, 1, 10);
    const density = computeKDE([0.5], grid);
    density.forEach((d) => expect(d).toBeGreaterThanOrEqual(0));
  });
});

describe('kdeIntersectionArea', () => {
  it('returns 0 for non-overlapping distributions', () => {
    const grid = linspace(0, 1, 5);
    const d1 = [1, 0, 0, 0, 0];
    const d2 = [0, 0, 0, 0, 1];
    expect(kdeIntersectionArea(d1, d2, grid)).toBeCloseTo(0);
  });
  it('returns area for identical distributions', () => {
    const grid = linspace(0, 1, 100);
    const d = grid.map(() => 1.0);
    const area = kdeIntersectionArea(d, d, grid);
    expect(area).toBeGreaterThan(0);
  });
});
