import { describe, it, expect } from 'vitest';
import { fibonacciSphere, marginToColor, computeLayout } from '../lib/graph-layout';

describe('fibonacciSphere', () => {
  it('returns n points', () => {
    expect(fibonacciSphere(10)).toHaveLength(10);
  });
  it('points are on sphere surface (approx)', () => {
    const pts = fibonacciSphere(5, 3);
    pts.forEach((p) => {
      const r = Math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2);
      expect(r).toBeCloseTo(3, 0);
    });
  });
});

describe('marginToColor', () => {
  it('returns green for strong favorite', () => expect(marginToColor(15)).toBe(0x00ff88));
  it('returns cyan for moderate favorite', () => expect(marginToColor(5)).toBe(0x00f5ff));
  it('returns amber for toss-up', () => expect(marginToColor(0)).toBe(0xffb800));
  it('returns violet for underdog', () => expect(marginToColor(-5)).toBe(0x7b2fff));
  it('returns red for heavy underdog', () => expect(marginToColor(-15)).toBe(0xff2d55));
});

describe('computeLayout', () => {
  it('returns conference and team positions', () => {
    const result = computeLayout(
      ['ACC', 'SEC'],
      { ACC: ['Duke', 'UNC'], SEC: ['Kentucky', 'Tennessee'] },
    );
    expect(result.conferencePositions).toHaveLength(2);
    expect(result.teamPositions).toHaveLength(4);
  });
});
