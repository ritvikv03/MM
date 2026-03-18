import { z } from 'zod';

export const TeamNodeSchema = z.object({
  id: z.string(),
  name: z.string(),
  conference: z.string(),
  seed: z.number().nullable(),
  adj_oe: z.number(),
  adj_de: z.number(),
  tempo: z.number(),
  x: z.number(),
  y: z.number(),
  z: z.number(),
  color: z.number(),
});

export const ConferenceNodeSchema = z.object({
  id: z.string(),
  name: z.string(),
  x: z.number(),
  y: z.number(),
  z: z.number(),
  color: z.number(),
});

export const GameEdgeSchema = z.object({
  source: z.string(),
  target: z.string(),
  home_win: z.boolean().nullable(),
  spread: z.number().nullable(),
  date: z.string().nullable(),
});

export const ConferenceEdgeSchema = z.object({
  source: z.string(),
  target: z.string(),
  edge_type: z.literal('member_of'),
});

export const GraphResponseSchema = z.object({
  teams: z.array(TeamNodeSchema),
  conferences: z.array(ConferenceNodeSchema),
  games: z.array(GameEdgeSchema),
  conference_edges: z.array(ConferenceEdgeSchema),
  data_source: z.enum(["real", "stub"]).default("stub"),
});

export const MatchupRequestSchema = z.object({
  home_team: z.string(),
  away_team: z.string(),
  season: z.number().default(2024),
  neutral_site: z.boolean().default(false),
});

export const MatchupResponseSchema = z.object({
  home_team: z.string(),
  away_team: z.string(),
  p_win_home: z.number(),
  p_win_samples: z.array(z.number()),
  spread_mean: z.number(),
  spread_samples: z.array(z.number()),
  luck_compressed: z.boolean(),
  data_source: z.enum(["real", "stub"]).default("stub"),
});

export const SimulateResponseSchema = z.object({
  n_simulations: z.number(),
  advancements: z.array(
    z.object({
      team: z.string(),
      advancement_probs: z.record(z.number()),
      entropy: z.number(),
    }),
  ),
  data_source: z.enum(["real", "stub"]).default("stub"),
});

// Enriched matchup response with factors, odds, and upset probability
export interface EnrichedMatchupResponse {
  home_team: string;
  away_team: string;
  p_win_home: number;
  p_win_samples: number[];
  spread_mean: number;
  spread_samples: number[];
  luck_compressed: boolean;
  // Enriched fields
  home_moneyline: string;
  away_moneyline: string;
  upset_probability: number;
  home_factors: string[];
  away_factors: string[];
  home_record: string;
  away_record: string;
  home_seed: number;
  away_seed: number;
  home_conference: string;
  away_conference: string;
  home_adj_oe: number;
  away_adj_oe: number;
  home_adj_de: number;
  away_adj_de: number;
  home_tempo: number;
  away_tempo: number;
  home_key_player: string;
  away_key_player: string;
}

export type TeamNode = z.infer<typeof TeamNodeSchema>;
export type ConferenceNode = z.infer<typeof ConferenceNodeSchema>;
export type GameEdge = z.infer<typeof GameEdgeSchema>;
export type ConferenceEdge = z.infer<typeof ConferenceEdgeSchema>;
export type GraphResponse = z.infer<typeof GraphResponseSchema>;
export type MatchupRequest = z.infer<typeof MatchupRequestSchema>;
export type MatchupResponse = z.infer<typeof MatchupResponseSchema>;
export type SimulateResponse = z.infer<typeof SimulateResponseSchema>;

// ── Intel ─────────────────────────────────────────────────────────────────────

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

export interface CinderellaEntry {
  team: string;
  seed: number;
  region: string;
  opponent: string;
  opponent_seed: number;
  upset_pct: number;
  edge_summary: string;
}

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

export interface FalseFavorite {
  team: string;
  seed: number;
  region: string;
  seed_label: string;
  risk_level: 'EXTREME' | 'HIGH' | 'MODERATE';
  detail: string;
  em: number;
  luck: number;
  adj_de: number;
}

export interface IntelResponse {
  season: number;
  flags: IntelFlag[];
  false_favorites: FalseFavorite[];
  cinderellas: CinderellaEntry[];
  deep_dives: MatchupDeepDive[];
  optimal_path: {
    r64_differentiators: Array<{
      winner: string;
      loser: string;
      winner_seed: number;
      loser_seed: number;
      region: string;
      upset_pct: number;
    }>;
    deep_run_value: Array<{
      team: string;
      seed: number;
      region: string;
      em: number;
      adj_oe: number;
      adj_de: number;
    }>;
    championship_edge: {
      team_a: string;
      team_b: string;
      p_a: number;
      p_b: number;
      note: string;
    } | null;
  };
}

export interface OptimalBracketResponse {
  champion: string;
  final_four: string[];
  n_simulations: number;
  advancements: Array<{
    team: string;
    advancement_probs: Record<string, number>;
    entropy: number;
  }>;
  data_source: string;
}
