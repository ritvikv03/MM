'use client';
import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { MOCK_TEAMS, getConferenceName } from '@/lib/mock-data';
import { BasketballCourt } from '@/components/ui/BasketballCourt';

export function PowerRankings() {
  const ranked = useMemo(() => {
    return [...MOCK_TEAMS]
      .map((t) => ({ ...t, netRating: t.adj_oe - t.adj_de }))
      .sort((a, b) => b.netRating - a.netRating);
  }, []);

  const maxOE = Math.max(...ranked.map(t => t.adj_oe));
  const minDE = Math.min(...ranked.map(t => t.adj_de));
  const maxNet = Math.max(...ranked.map(t => t.netRating));

  return (
    <div className="p-5 max-w-5xl mx-auto relative">
      {/* Court watermark */}
      <div style={{ position: 'absolute', top: '10%', right: '-5%', width: '300px', opacity: 0.04, pointerEvents: 'none', transform: 'rotate(90deg)' }}>
        <BasketballCourt />
      </div>

      <div className="flex items-center gap-3 mb-6">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ff6b35" strokeWidth="1.5">
          <rect x="3" y="14" width="4" height="7" rx="1" />
          <rect x="10" y="8" width="4" height="13" rx="1" />
          <rect x="17" y="3" width="4" height="18" rx="1" />
        </svg>
        <h1 style={{ fontFamily: 'var(--font-space-grotesk)', color: '#ff6b35', fontWeight: 700, fontSize: '20px', letterSpacing: '0.05em' }}>
          POWER RANKINGS
        </h1>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
          ST-GNN Model • {ranked.length} Teams
        </span>
      </div>

      {/* Column headers */}
      <div className="glass-wood rounded-lg mb-2 px-4 py-2" style={{ display: 'grid', gridTemplateColumns: '40px 36px 1fr 80px 100px 100px 80px', gap: '8px', alignItems: 'center' }}>
        <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600 }}>RK</span>
        <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600 }}>SEED</span>
        <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600 }}>TEAM</span>
        <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600 }}>RECORD</span>
        <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600, textAlign: 'center' }}>OFF EFF</span>
        <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600, textAlign: 'center' }}>DEF EFF</span>
        <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600, textAlign: 'right' }}>NET</span>
      </div>

      {/* Team rows */}
      {ranked.map((team, i) => (
        <motion.div
          key={team.name}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: i * 0.03 }}
          className="glass-wood rounded-lg mb-1.5 px-4 py-3"
          style={{
            display: 'grid',
            gridTemplateColumns: '40px 36px 1fr 80px 100px 100px 80px',
            gap: '8px',
            alignItems: 'center',
            borderLeft: i < 4 ? '3px solid #ff6b35' : i < 8 ? '3px solid #d4a843' : '3px solid transparent',
          }}
        >
          {/* Rank */}
          <span style={{ fontSize: '16px', fontWeight: 700, color: i < 4 ? '#ff6b35' : i < 8 ? '#d4a843' : 'var(--text-secondary)' }}>
            {i + 1}
          </span>

          {/* Seed */}
          <div style={{
            width: '28px', height: '28px', borderRadius: '50%',
            background: 'rgba(255, 107, 53, 0.15)',
            border: '1px solid rgba(255, 107, 53, 0.3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '11px', fontWeight: 700, color: '#ff6b35',
          }}>
            {team.seed}
          </div>

          {/* Team + Conference */}
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>{team.name}</div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{getConferenceName(team.conference)}</div>
          </div>

          {/* Record */}
          <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{team.record}</span>

          {/* Offensive Efficiency */}
          <div>
            <div style={{ fontSize: '11px', color: '#2ecc71', fontWeight: 600, marginBottom: '3px', textAlign: 'center' }}>
              {team.adj_oe.toFixed(1)}
            </div>
            <div className="stat-bar">
              <div className="stat-bar-fill" style={{ width: `${(team.adj_oe / maxOE) * 100}%`, background: 'linear-gradient(90deg, rgba(46,204,113,0.5), #2ecc71)' }} />
            </div>
          </div>

          {/* Defensive Efficiency */}
          <div>
            <div style={{ fontSize: '11px', color: '#3498db', fontWeight: 600, marginBottom: '3px', textAlign: 'center' }}>
              {team.adj_de.toFixed(1)}
            </div>
            <div className="stat-bar">
              <div className="stat-bar-fill" style={{ width: `${(minDE / team.adj_de) * 100}%`, background: 'linear-gradient(90deg, rgba(52,152,219,0.5), #3498db)' }} />
            </div>
          </div>

          {/* Net Rating */}
          <div style={{ textAlign: 'right' }}>
            <span style={{
              fontSize: '14px', fontWeight: 700,
              color: team.netRating > 25 ? '#2ecc71' : team.netRating > 20 ? '#d4a843' : 'var(--text-secondary)',
            }}>
              +{team.netRating.toFixed(1)}
            </span>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
