/**
 * frontend/lib/queries.ts
 * SWR hooks that read pre-computed pipeline outputs from Supabase.
 *
 * Data flow (no backend server required):
 *   GitHub Actions 3×/day → computes predictions → writes to Supabase
 *   Netlify frontend → reads from Supabase directly via these hooks
 */
import useSWR from 'swr';
import { supabase } from './supabase';
import type { IntelResponse } from './api';

const POLL_MS = 5 * 60 * 1000; // 5-minute re-fetch

// ── Types ────────────────────────────────────────────────────────────────────

export interface TeamRow {
  name: string;
  seed: number;
  conference: string;
  adj_oe: number;
  adj_de: number;
  adj_em: number;
  tempo: number;
  luck: number;
  sos: number;
  coach: string;
  region: string;
  season: number;
}

export interface BracketRunRow {
  champion: string;
  final_four: string[];
  advancements: {
    team: string;
    advancement_probs: Record<string, number>;
    champ_probability: number;
    entropy: number;
  }[];
  n_simulations: number;
  computed_at: string;
}

// ── Fetchers ─────────────────────────────────────────────────────────────────

async function fetchTeams(season: number): Promise<TeamRow[]> {
  if (!supabase) return [];
  const { data, error } = await supabase
    .from('teams')
    .select('name,seed,conference,adj_oe,adj_de,adj_em,tempo,luck,sos,coach,region,season')
    .eq('season', season)
    .order('adj_em', { ascending: false });
  if (error) throw error;
  return (data ?? []) as TeamRow[];
}

async function fetchLatestBracketRun(season: number): Promise<BracketRunRow | null> {
  if (!supabase) return null;
  const { data, error } = await supabase
    .from('bracket_runs')
    .select('advancement_probs,champion_prob,n_simulations,computed_at')
    .eq('season', season)
    .order('computed_at', { ascending: false })
    .limit(1)
    .single();
  if (error || !data) return null;

  // Flatten advancement_probs JSONB into the expected shape
  const adv = data.advancement_probs as Record<string, Record<string, number>>;
  const champ = data.champion_prob as Record<string, number>;
  const advancements = Object.entries(adv ?? {}).map(([team, probs]) => ({
    team,
    advancement_probs: probs,
    champ_probability: champ?.[team] ?? probs?.Championship ?? 0,
    entropy: 0,
  }));
  const sortedByChamp = [...advancements].sort((a, b) => b.champ_probability - a.champ_probability);
  const champion = sortedByChamp[0]?.team ?? '';
  const final_four = sortedByChamp
    .sort((a, b) => (b.advancement_probs?.F4 ?? 0) - (a.advancement_probs?.F4 ?? 0))
    .slice(0, 4)
    .map(t => t.team);

  return {
    champion,
    final_four,
    advancements,
    n_simulations: data.n_simulations ?? 0,
    computed_at: data.computed_at,
  };
}

async function fetchLatestIntel(season: number): Promise<IntelResponse | null> {
  if (!supabase) return null;
  const { data, error } = await supabase
    .from('intel_snapshots')
    .select('snapshot')
    .eq('season', season)
    .order('computed_at', { ascending: false })
    .limit(1)
    .single();
  if (error || !data) return null;
  return data.snapshot as IntelResponse;
}

// ── SWR Hooks ─────────────────────────────────────────────────────────────────

export function useTeams(season = 2026) {
  const { data, error, isLoading } = useSWR(
    ['teams', season],
    () => fetchTeams(season),
    { refreshInterval: POLL_MS, revalidateOnFocus: false },
  );
  return { teams: data ?? [], isLoading, error };
}

export function useBracketOptimal(season = 2026) {
  const { data, error, isLoading } = useSWR(
    ['bracket_optimal', season],
    () => fetchLatestBracketRun(season),
    { refreshInterval: POLL_MS, revalidateOnFocus: false },
  );
  return { bracketRun: data ?? null, isLoading, error };
}

export function useIntelSnapshot(season = 2026) {
  const { data, error, isLoading } = useSWR(
    ['intel_snapshot', season],
    () => fetchLatestIntel(season),
    { refreshInterval: POLL_MS, revalidateOnFocus: false },
  );
  return { intel: data ?? null, isLoading, error };
}
