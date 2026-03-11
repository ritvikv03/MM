import { describe, it, expect } from 'vitest';
import { COLORS, BLOOM, hexToRgb } from '../lib/colors';

describe('COLORS', () => {
  it('has void_black', () => expect(COLORS.void_black).toBe('#020408'));
  it('has cyan_core', () => expect(COLORS.cyan_core).toBe('#00f5ff'));
  it('has violet_deep', () => expect(COLORS.violet_deep).toBe('#7b2fff'));
  it('has tritium_green', () => expect(COLORS.tritium_green).toBe('#00ff88'));
});

describe('hexToRgb', () => {
  it('parses #00f5ff', () => {
    const rgb = hexToRgb('#00f5ff');
    expect(rgb.r).toBe(0);
    expect(rgb.g).toBe(245);
    expect(rgb.b).toBe(255);
  });
  it('throws on invalid hex', () => {
    expect(() => hexToRgb('not-a-color')).toThrow();
  });
});

describe('BLOOM', () => {
  it('has cyan bloom config', () => {
    expect(BLOOM.cyan.intensity).toBeGreaterThan(0);
    expect(BLOOM.cyan.color).toBe(COLORS.cyan_core);
  });
});
