/**
 * frontend/lib/hooks/use-live-data.ts
 * SWR-based hooks for all live backend data.
 *
 * All components must consume data through these hooks — no direct fetch calls.
 * Polling interval: 5 minutes for intel + graph; on-demand for bracket/matchup.
 */

import useSWR, { SWRConfiguration } from 'swr';
import {
  fetchGraph,
  fetchIntel,
  fetchOptimalBracket,
  type IntelResponse,
  type OptimalBracketResponse,
} from '@/lib/api';
import type { GraphResponse } from '@/lib/api-types';

const POLL_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

const defaultConfig: SWRConfiguration = {
  revalidateOnFocus: false,
  dedupingInterval: 60_000,
  errorRetryCount: 3,
  errorRetryInterval: 10_000,
};

// ─── Graph hook ────────────────────────────────────────────────────────────────
export function useGraph(season: number = 2026) {
  const { data, error, isLoading, mutate } = useSWR<GraphResponse>(
    ['graph', season],
    () => fetchGraph(season),
    {
      ...defaultConfig,
      refreshInterval: POLL_INTERVAL_MS,
    }
  );
  return { data: data ?? null, error, isLoading, mutate };
}

// ─── Intel hook ────────────────────────────────────────────────────────────────
export function useIntel(season: number = 2026) {
  const { data, error, isLoading, mutate } = useSWR<IntelResponse>(
    ['intel', season],
    () => fetchIntel(season),
    {
      ...defaultConfig,
      refreshInterval: POLL_INTERVAL_MS,
    }
  );
  return { data: data ?? null, error, isLoading, mutate };
}

// ─── Optimal bracket hook ──────────────────────────────────────────────────────
export function useOptimalBracket() {
  const { data, error, isLoading, mutate } = useSWR<OptimalBracketResponse>(
    'optimal-bracket',
    fetchOptimalBracket,
    {
      ...defaultConfig,
      // Don't auto-poll — user-triggered or once per session
      refreshInterval: 0,
    }
  );
  return { data: data ?? null, error, isLoading, mutate };
}
