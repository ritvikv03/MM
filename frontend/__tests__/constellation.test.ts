import { describe, it, expect } from 'vitest';
import { fibonacciSphere, marginToColor, computeLayout } from '../lib/graph-layout';
import { COLORS } from '../lib/colors';

describe('ConstellationCanvas data helpers', () => {
  it('fibonacciSphere returns correct count', () => {
    expect(fibonacciSphere(32)).toHaveLength(32);
  });

  it('fibonacciSphere points lie approximately on sphere', () => {
    const pts = fibonacciSphere(10, 5);
    pts.forEach((p) => {
      const r = Math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2);
      expect(r).toBeCloseTo(5, 0);
    });
  });

  it('marginToColor maps positive margin to green', () => {
    expect(marginToColor(15)).toBe(0x00ff88);
  });

  it('marginToColor maps zero to amber', () => {
    expect(marginToColor(0)).toBe(0xffb800);
  });

  it('marginToColor maps large negative to red', () => {
    expect(marginToColor(-20)).toBe(0xff2d55);
  });

  it('computeLayout positions conferences at fibonacci sphere', () => {
    const { conferencePositions } = computeLayout(
      ['ACC', 'SEC', 'Big 12'],
      { ACC: ['Duke'], SEC: ['Kentucky'], 'Big 12': ['Kansas'] },
    );
    expect(conferencePositions).toHaveLength(3);
    conferencePositions.forEach((c) => {
      const r = Math.sqrt(c.x ** 2 + c.y ** 2 + c.z ** 2);
      expect(r).toBeGreaterThan(0);
    });
  });

  it('computeLayout places teams near their conference center', () => {
    const { conferencePositions, teamPositions } = computeLayout(
      ['ACC'],
      { ACC: ['Duke', 'UNC', 'NC State'] },
      5,
      1.5,
    );
    const conf = conferencePositions[0];
    teamPositions.forEach((t) => {
      const dx = t.x - conf.x;
      const dy = t.y - conf.y;
      const dz = t.z - conf.z;
      const dist = Math.sqrt(dx ** 2 + dy ** 2 + dz ** 2);
      expect(dist).toBeLessThan(5); // within teamSpread radius
    });
  });

  it('COLORS token has correct void_black', () => {
    expect(COLORS.void_black).toBe('#020408');
  });
});
