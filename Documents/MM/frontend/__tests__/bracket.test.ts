import { describe, it, expect } from 'vitest';
import {
  getAdvancementProb,
  rankByRound,
  probToHeatColor,
  normalizeEntropy,
  topChampionshipContenders,
  ROUNDS,
} from '../lib/bracket-utils';
import type { TeamAdvancement } from '../lib/bracket-utils';

const TEAMS: TeamAdvancement[] = [
  {
    team: 'Duke',
    advancement_probs: { R64: 0.9, R32: 0.75, S16: 0.55, E8: 0.35, F4: 0.2, Championship: 0.1 },
    entropy: 1.8,
  },
  {
    team: 'UNC',
    advancement_probs: { R64: 0.85, R32: 0.65, S16: 0.4, E8: 0.2, F4: 0.08, Championship: 0.04 },
    entropy: 1.5,
  },
  {
    team: 'Kansas',
    advancement_probs: { R64: 0.92, R32: 0.8, S16: 0.6, E8: 0.4, F4: 0.25, Championship: 0.15 },
    entropy: 1.9,
  },
];

describe('getAdvancementProb', () => {
  it('returns correct probability for known round', () => {
    expect(getAdvancementProb(TEAMS[0], 'Championship')).toBe(0.1);
  });
  it('returns 0 for unknown round', () => {
    expect(getAdvancementProb(TEAMS[0], 'R64')).toBe(0.9);
  });
});

describe('rankByRound', () => {
  it('sorts by championship probability descending', () => {
    const ranked = rankByRound(TEAMS, 'Championship');
    expect(ranked[0].team).toBe('Kansas');
    expect(ranked[1].team).toBe('Duke');
    expect(ranked[2].team).toBe('UNC');
  });
  it('does not mutate original array', () => {
    const original = [...TEAMS];
    rankByRound(TEAMS, 'F4');
    expect(TEAMS).toEqual(original);
  });
});

describe('probToHeatColor', () => {
  it('returns a CSS rgb string', () => {
    const c = probToHeatColor(0.5);
    expect(c).toMatch(/^rgb\(\d+,\d+,\d+\)$/);
  });
  it('clamps values above 1', () => {
    expect(probToHeatColor(2.0)).toBe(probToHeatColor(1.0));
  });
  it('clamps values below 0', () => {
    expect(probToHeatColor(-1.0)).toBe(probToHeatColor(0.0));
  });
});

describe('normalizeEntropy', () => {
  it('normalizes to 0 for entropy=0', () => {
    expect(normalizeEntropy(0)).toBe(0);
  });
  it('normalizes to 1 for max entropy', () => {
    const maxH = Math.log2(6);
    expect(normalizeEntropy(maxH, 6)).toBeCloseTo(1.0);
  });
  it('clamps above 1', () => {
    expect(normalizeEntropy(1000, 6)).toBe(1);
  });
});

describe('topChampionshipContenders', () => {
  it('returns top N teams', () => {
    const top = topChampionshipContenders(TEAMS, 2);
    expect(top).toHaveLength(2);
    expect(top[0].team).toBe('Kansas');
  });
  it('returns all if n > teams length', () => {
    const top = topChampionshipContenders(TEAMS, 10);
    expect(top).toHaveLength(3);
  });
});

describe('ROUNDS constant', () => {
  it('has 6 rounds', () => {
    expect(ROUNDS).toHaveLength(6);
  });
  it('starts with R64', () => {
    expect(ROUNDS[0]).toBe('R64');
  });
  it('ends with Championship', () => {
    expect(ROUNDS[ROUNDS.length - 1]).toBe('Championship');
  });
});
