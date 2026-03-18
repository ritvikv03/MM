'use client';
import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { useTeams } from '@/lib/queries';
import { TOURNAMENT_TEAMS_2026 as FALLBACK_TEAMS } from '@/lib/team-data';

// Typical public pick % by seed derived from ESPN/Yahoo bracket data.
// Lower seeds attract far more public picks regardless of true probability.
const SEED_PUBLIC_PCT: Record<number, number> = {
  1: 0.72, 2: 0.58, 3: 0.48, 4: 0.40, 5: 0.35, 6: 0.30,
  7: 0.25, 8: 0.20, 9: 0.18, 10: 0.15, 11: 0.12, 12: 0.10,
  13: 0.06, 14: 0.04, 15: 0.02, 16: 0.01,
};

// Deterministic leverage from T-Rank data — no random, no hardcoding.
// True win prob proxy = sigmoid(adj_em / 8); public ownership from seed lookup.
function getLeverageScore(adjEm: number, luck: number, seed: number): number {
  const em = adjEm - luck * 2.5; // luck-adjusted EM
  const trueWinProb = 1 / (1 + Math.exp(-em / 8));
  const publicPct = SEED_PUBLIC_PCT[seed] ?? 0.15;
  return Math.round((trueWinProb / publicPct) * 100) / 100;
}

function getColorForLeverage(score: number): string {
  if (score >= 1.2) return '#2ecc71'; // Green: High Value
  if (score <= 0.85) return '#e74c3c'; // Red: Toxic Chalk
  return '#d4a843'; // Yellow: Fair Value
}

export function WarRoomMatrix() {
  const [filter, setFilter] = useState<'all' | 'value' | 'toxic'>('all');
  const { teams: liveTeams, isLoading } = useTeams(2026);

  const teamsWithLeverage = useMemo(() => {
    const source = liveTeams.length > 0
      ? liveTeams.map(t => ({
          name: t.name, seed: t.seed, region: t.region ?? '',
          adj_oe: t.adj_oe, adj_de: t.adj_de, adj_em: t.adj_em ?? (t.adj_oe - t.adj_de),
          luck: t.luck ?? 0, playStyle: '',
        }))
      : FALLBACK_TEAMS.map(t => ({
          name: t.name, seed: t.seed, region: t.region ?? '',
          adj_oe: t.adj_oe, adj_de: t.adj_de, adj_em: t.adj_oe - t.adj_de,
          luck: t.luck, playStyle: t.playStyle ?? '',
        }));
    return source.map(team => ({
      ...team,
      leverage: getLeverageScore(team.adj_em, team.luck, team.seed),
    })).sort((a, b) => b.leverage - a.leverage);
  }, [liveTeams]);

  const filteredTeams = useMemo(() => {
    if (filter === 'value') return teamsWithLeverage.filter(t => t.leverage >= 1.2);
    if (filter === 'toxic') return teamsWithLeverage.filter(t => t.leverage <= 0.85);
    return teamsWithLeverage;
  }, [teamsWithLeverage, filter]);

  return (
    <div className="p-5 max-w-6xl mx-auto flex flex-col h-full gap-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" strokeWidth="1.5" strokeLinecap="round">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
          </svg>
          <h1 style={{ fontFamily: 'var(--font-space-grotesk)', color: '#2ecc71', fontWeight: 700, fontSize: '20px', letterSpacing: '0.05em' }}>
            WAR ROOM MATRIX
          </h1>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginLeft: '8px' }}>
            {liveTeams.length > 0 ? 'Live T-Rank' : isLoading ? 'Loading…' : 'Cached'}
          </span>
        </div>
        
        <div className="flex gap-2">
          {['all', 'value', 'toxic'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f as any)}
              className="px-3 py-1 rounded"
              style={{
                fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em',
                background: filter === f ? 'rgba(46, 204, 113, 0.15)' : 'rgba(255,107,53,0.05)',
                color: filter === f ? '#2ecc71' : 'var(--text-muted)',
                border: `1px solid ${filter === f ? '#2ecc71' : 'transparent'}`
              }}
            >
              {f === 'toxic' ? 'Toxic Chalk' : f === 'value' ? 'High Leverage' : 'All Teams'}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-highlight p-4 rounded-lg mb-2" style={{ borderLeft: '3px solid #2ecc71' }}>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
          <strong>Leverage Score = (True Win Prob / Public Pick %).</strong> <br/>
          Identify market inefficiencies. <span style={{ color: '#2ecc71' }}>Green</span> indicates under-owned value. <span style={{ color: '#e74c3c' }}>Red</span> indicates over-owned "toxic chalk" to fade in large bracket contests (competitions where many people submit brackets).
        </p>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
        gap: '12px',
        overflow: 'auto',
        paddingRight: '4px'
      }}>
        {filteredTeams.map((team, index) => (
          <motion.div
            key={team.name}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.01 }}
            className="glass-wood p-3 rounded-lg flex flex-col gap-2"
            style={{ borderTop: `2px solid ${getColorForLeverage(team.leverage)}` }}
          >
            <div className="flex justify-between items-start">
              <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600 }}>{team.region} #{team.seed}</span>
              <span style={{ 
                fontSize: '14px', 
                fontWeight: 800, 
                color: getColorForLeverage(team.leverage) 
              }}>
                {team.leverage.toFixed(2)}x
              </span>
            </div>
            <div>
              <div style={{ fontSize: '15px', fontWeight: 700, color: '#ff6b35' }}>{team.name}</div>
              <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>{team.playStyle}</div>
            </div>
            
            <div className="mt-2 pt-2" style={{ borderTop: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between' }}>
              <div className="flex flex-col text-center">
                <span style={{ fontSize: '8px', color: 'var(--text-muted)' }}>ADJ EM</span>
                <span style={{ fontSize: '11px', color: '#d4a843', fontWeight: 600 }}>+{team.adj_em.toFixed(1)}</span>
              </div>
              <div className="flex flex-col text-center">
                <span style={{ fontSize: '8px', color: 'var(--text-muted)' }}>LUCK</span>
                <span style={{ fontSize: '11px', color: team.luck > 0 ? '#e74c3c' : '#2ecc71', fontWeight: 600 }}>{team.luck > 0 ? '+' : ''}{team.luck.toFixed(2)}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
