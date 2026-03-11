import { describe, it, expect } from 'vitest';
import { TeamNodeSchema, GraphResponseSchema, MatchupResponseSchema } from '../lib/api-types';

describe('TeamNodeSchema', () => {
  it('parses valid team node', () => {
    const node = TeamNodeSchema.parse({
      id: 'duke',
      name: 'Duke',
      conference: 'ACC',
      seed: 1,
      adj_oe: 120,
      adj_de: 95,
      tempo: 68,
      x: 1.0,
      y: 2.0,
      z: 3.0,
      color: 0x00f5ff,
    });
    expect(node.name).toBe('Duke');
  });
  it('allows null seed', () => {
    const node = TeamNodeSchema.parse({
      id: 'duke',
      name: 'Duke',
      conference: 'ACC',
      seed: null,
      adj_oe: 120,
      adj_de: 95,
      tempo: 68,
      x: 0,
      y: 0,
      z: 0,
      color: 0,
    });
    expect(node.seed).toBeNull();
  });
});

describe('MatchupResponseSchema', () => {
  it('parses valid matchup response', () => {
    const resp = MatchupResponseSchema.parse({
      home_team: 'Duke',
      away_team: 'UNC',
      p_win_home: 0.65,
      p_win_samples: [0.6, 0.7],
      spread_mean: -4.5,
      spread_samples: [-5, -4],
      luck_compressed: false,
    });
    expect(resp.p_win_home).toBe(0.65);
    expect(resp.luck_compressed).toBe(false);
  });
});
