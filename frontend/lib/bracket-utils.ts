// bracket-utils.ts — shared constants and helpers for bracket components

export const ROUNDS = ['R64', 'R32', 'S16', 'E8', 'F4', 'Championship'];

export interface TeamAdvancement {
  team: string;
  advancement_probs: Record<string, number>;
  entropy: number;
}

/**
 * Convert a probability (0..1) to a heat-map RGB color string.
 * Returns `rgb(r,g,b)` with clamped input.
 */
export function probToHeatColor(prob: number): string {
  const p = Math.max(0, Math.min(1, prob));
  if (p >= 0.7) return `rgb(46,204,113)`;   // emerald
  if (p >= 0.5) return `rgb(212,168,67)`;   // gold
  if (p >= 0.3) return `rgb(52,152,219)`;   // blue
  if (p >= 0.15) return `rgb(149,165,166)`; // grey
  return `rgb(80,90,91)`;                   // faded
}

/**
 * Get the advancement probability for a specific round.
 */
export function getAdvancementProb(team: TeamAdvancement, round: string): number {
  return team.advancement_probs[round] ?? 0;
}

/**
 * Sort teams by their advancement probability in a given round (descending).
 * Returns a new array; does not mutate the input.
 */
export function rankByRound(teams: TeamAdvancement[], round: string): TeamAdvancement[] {
  return [...teams].sort((a, b) => getAdvancementProb(b, round) - getAdvancementProb(a, round));
}

/**
 * Normalize Shannon entropy to [0, 1] range.
 * maxRounds defaults to the length of ROUNDS (6).
 * Based on H_max = log2(maxRounds).
 */
export function normalizeEntropy(h: number, maxRounds: number = ROUNDS.length): number {
  const hMax = Math.log2(maxRounds);
  if (hMax === 0) return 0;
  return Math.max(0, Math.min(1, h / hMax));
}

/**
 * Return the top N teams by Championship probability.
 */
export function topChampionshipContenders(teams: TeamAdvancement[], n: number): TeamAdvancement[] {
  return rankByRound(teams, 'Championship').slice(0, n);
}
