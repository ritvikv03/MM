import {
  GraphResponseSchema,
  GraphResponse,
  MatchupRequest,
  SimulateResponseSchema,
  SimulateResponse,
} from './api-types';
import type { EnrichedMatchupResponse } from './api-types';
import { getMockGraph, getMockMatchup, getMockSimulation } from './mock-data';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export async function fetchGraph(season: number = 2024): Promise<GraphResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/graph?season=${season}`);
    if (!res.ok) throw new Error(`Graph fetch failed: ${res.status}`);
    return GraphResponseSchema.parse(await res.json());
  } catch (err) {
    console.error('[api] fetchGraph failed, using mock:', err);
    return getMockGraph();
  }
}

export async function fetchMatchup(req: MatchupRequest): Promise<EnrichedMatchupResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/matchup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`Matchup fetch failed: ${res.status}`);
    const real = await res.json();
    // Merge real API data with mock enrichment fields; real data takes priority
    const mock = getMockMatchup(req.home_team, req.away_team);
    return {
      ...mock,
      p_win_home: real.p_win_home ?? mock.p_win_home,
      p_win_samples: real.p_win_samples ?? mock.p_win_samples,
      spread_mean: real.spread_mean ?? mock.spread_mean,
      spread_samples: real.spread_samples ?? mock.spread_samples,
      luck_compressed: real.luck_compressed ?? mock.luck_compressed,
    };
  } catch (err) {
    console.error('[api] fetchMatchup failed, using mock:', err);
    return getMockMatchup(req.home_team, req.away_team);
  }
}

export async function simulateBracket(
  teams: string[],
  n_simulations: number = 1000,
): Promise<SimulateResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/bracket/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ teams, n_simulations }),
    });
    if (!res.ok) throw new Error(`Simulate fetch failed: ${res.status}`);
    return SimulateResponseSchema.parse(await res.json());
  } catch (err) {
    console.error('[api] simulateBracket failed, using mock:', err);
    return getMockSimulation(teams, n_simulations);
  }
}
