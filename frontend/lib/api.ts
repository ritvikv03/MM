import {
  GraphResponseSchema,
  GraphResponse,
  MatchupRequest,
  SimulateResponseSchema,
  SimulateResponse,
} from './api-types';
import type { EnrichedMatchupResponse } from './api-types';
import { TOURNAMENT_TEAMS_2026 } from './team-data';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export async function fetchGraph(season: number = 2026): Promise<GraphResponse> {
  const res = await fetch(`${API_BASE}/api/graph?season=${season}`);
  if (!res.ok) throw new Error(`Graph fetch failed: ${res.status}`);
  return GraphResponseSchema.parse(await res.json());
}

export async function fetchMatchup(req: MatchupRequest): Promise<EnrichedMatchupResponse> {
  // Build enrichment from local team data
  const homeData = TOURNAMENT_TEAMS_2026.find(t => t.name === req.home_team);
  const awayData = TOURNAMENT_TEAMS_2026.find(t => t.name === req.away_team);

  const fallback: EnrichedMatchupResponse = {
    home_team: req.home_team,
    away_team: req.away_team,
    p_win_home: 0.5,
    p_win_samples: [],
    spread_mean: 0,
    spread_samples: [],
    luck_compressed: false,
    home_moneyline: '-110',
    away_moneyline: '-110',
    upset_probability: 0.0,
    home_factors: ['Insufficient data'],
    away_factors: ['Insufficient data'],
    home_record: homeData?.record ?? '0-0',
    away_record: awayData?.record ?? '0-0',
    home_seed: homeData?.seed ?? 16,
    away_seed: awayData?.seed ?? 16,
    home_conference: homeData?.conference ?? '',
    away_conference: awayData?.conference ?? '',
    home_adj_oe: homeData?.adj_oe ?? 100,
    away_adj_oe: awayData?.adj_oe ?? 100,
    home_adj_de: homeData?.adj_de ?? 100,
    away_adj_de: awayData?.adj_de ?? 100,
    home_tempo: homeData?.tempo ?? 68,
    away_tempo: awayData?.tempo ?? 68,
    home_key_player: homeData?.keyPlayer ?? 'TBD',
    away_key_player: awayData?.keyPlayer ?? 'TBD',
  };

  try {
    const res = await fetch(`${API_BASE}/api/matchup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...req, season: req.season ?? 2026 }),
    });
    if (!res.ok) throw new Error(`Matchup fetch failed: ${res.status}`);
    const real = await res.json();

    const pWin = real.p_win_home ?? fallback.p_win_home;
    const spreadMean = real.spread_mean ?? fallback.spread_mean;
    const upsetProb = pWin > 0.5
      ? 1 - pWin
      : pWin;

    // Generate moneylines from probability
    const toML = (p: number) => {
      if (p >= 0.5) return `-${Math.round(p / (1 - p) * 100)}`;
      return `+${Math.round((1 - p) / p * 100)}`;
    };

    // Generate analysis factors
    const homeEM = (homeData?.adj_oe ?? 100) - (homeData?.adj_de ?? 100);
    const awayEM = (awayData?.adj_oe ?? 100) - (awayData?.adj_de ?? 100);

    return {
      ...fallback,
      p_win_home: pWin,
      p_win_samples: real.p_win_samples ?? fallback.p_win_samples,
      spread_mean: spreadMean,
      spread_samples: real.spread_samples ?? fallback.spread_samples,
      luck_compressed: real.luck_compressed ?? fallback.luck_compressed,
      home_moneyline: toML(pWin),
      away_moneyline: toML(1 - pWin),
      upset_probability: upsetProb,
      home_factors: [
        `AdjEM: +${homeEM.toFixed(1)} (${homeEM > awayEM ? 'advantage' : 'disadvantage'})`,
        `Tempo: ${homeData?.tempo.toFixed(0)} poss/40min`,
        `Coach: ${homeData?.coachName} (${homeData?.coachTourneyRecord})`,
        homeData?.playStyle ?? '',
      ],
      away_factors: [
        `AdjEM: +${awayEM.toFixed(1)} (${awayEM > homeEM ? 'advantage' : 'disadvantage'})`,
        `Tempo: ${awayData?.tempo.toFixed(0)} poss/40min`,
        `Coach: ${awayData?.coachName} (${awayData?.coachTourneyRecord})`,
        awayData?.playStyle ?? '',
      ],
    };
  } catch (err) {
    console.error('[api] fetchMatchup failed:', err);
    return fallback;
  }
}

export async function simulateBracket(
  teams: string[],
  n_simulations: number = 1000,
): Promise<SimulateResponse> {
  const res = await fetch(`${API_BASE}/api/bracket/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ teams, n_simulations, season: 2026 }),
  });
  if (!res.ok) throw new Error(`Simulate fetch failed: ${res.status}`);
  return SimulateResponseSchema.parse(await res.json());
}

export async function fetchPredictedBracket(): Promise<{
  champion: string;
  final_four: string[];
  n_simulations: number;
  advancements: { team: string; advancement_probs: Record<string, number>; entropy: number }[];
  data_source: string;
}> {
  const res = await fetch(`${API_BASE}/api/bracket/predict`);
  if (!res.ok) throw new Error(`Predict fetch failed: ${res.status}`);
  return res.json();
}

/** Matches IntelFlag dataclass from intel_engine.py */
export interface IntelFlag {
  type: 'risk' | 'alert' | 'surge' | 'cinderella';
  severity: 'EXTREME' | 'HIGH' | 'MODERATE' | 'LOW';
  team: string;
  seed: number;
  region: string;
  headline: string;
  detail: string;
  metric: string;
  emoji: string;
}

/** Matches CinderellaEntry dataclass from intel_engine.py */
export interface CinderellaEntry {
  team: string;
  seed: number;
  region: string;
  opponent: string;
  opponent_seed: number;
  upset_pct: number;
  edge_summary: string;
}

/** Matches MatchupDeepDive dataclass from intel_engine.py */
export interface MatchupDeepDive {
  team_a: string;
  seed_a: number;
  team_b: string;
  seed_b: number;
  region: string;
  round: string;
  p_win_a: number;
  tempo_clash: boolean;
  tempo_diff: number;
  edge_team: string;
  em_delta: number;
  narrative: string;
  recommendation: string;
}

/** Matches IntelResponse dataclass from intel_engine.py */
export interface IntelResponse {
  season: number;
  flags: IntelFlag[];
  false_favorites: {
    team: string;
    seed: number;
    region: string;
    seed_label: string;
    risk_level: string;
    detail: string;
    em: number;
    luck: number;
    adj_de: number;
  }[];
  cinderellas: CinderellaEntry[];
  deep_dives: MatchupDeepDive[];
  optimal_path: {
    r64_differentiators: { winner: string; loser: string; winner_seed: number; loser_seed: number; region: string; upset_pct: number }[];
    deep_run_value: { team: string; seed: number; region: string; em: number; adj_oe: number; adj_de: number }[];
    championship_edge: { team_a: string; team_b: string; p_a: number; p_b: number; note: string } | null;
  };
}

export async function fetchIntel(season: number = 2026): Promise<IntelResponse> {
  const res = await fetch(`${API_BASE}/api/intel?season=${season}`);
  if (!res.ok) throw new Error(`Intel fetch failed: ${res.status}`);
  return res.json();
}

export interface OptimalBracketResponse {
  champion: string;
  final_four: string[];
  elite_eight: string[];
  n_simulations: number;
  advancements: {
    team: string;
    advancement_probs: Record<string, number>;
    entropy: number;
    champ_probability: number;
  }[];
  data_source: string;
}

export async function fetchOptimalBracket(): Promise<OptimalBracketResponse> {
  const res = await fetch(`${API_BASE}/api/bracket/optimal`);
  if (!res.ok) throw new Error(`Optimal bracket fetch failed: ${res.status}`);
  return res.json();
}
