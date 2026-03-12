'use client';
import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { MOCK_TEAMS } from '@/lib/mock-data';

// Helper: Pseudo-leverage score generator for the UI mockup
// Green = High Value (> 1.2), Red = Toxic Chalk (< 0.8), Yellow = Fair (0.8 - 1.2)
function getLeverageScore(seed: number, name: string): number {
  // Hardcode some narratives to match the Python engine logic
  if (name === 'Vanderbilt') return 0.65;  // Toxic chalk (collapsing defense)
  if (name === 'Purdue') return 0.75;      // Over-owned due to luck
  if (name === 'Vermont') return 1.45;     // Massive leverage upset
  if (name === 'Liberty') return 1.35;     // Massive leverage upset
  if (name === 'Yale') return 1.30;        // High leverage
  if (name === 'TCU') return 1.25;         // Coin-flip leverage
  
  if (seed === 1) return 0.95 + Math.random() * 0.1;
  if (seed >= 2 && seed <= 4) return 0.85 + Math.random() * 0.2;
  if (seed >= 5 && seed <= 7) return 0.80 + Math.random() * 0.3;
  if (seed >= 10 && seed <= 13) return 1.0 + Math.random() * 0.5; // Value zone
  
  return 0.9 + Math.random() * 0.2;
}

function getColorForLeverage(score: number): string {
  if (score >= 1.2) return '#2ecc71'; // Green: High Value
  if (score <= 0.85) return '#e74c3c'; // Red: Toxic Chalk
  return '#d4a843'; // Yellow: Fair Value
}

export function WarRoomMatrix() {
  const [filter, setFilter] = useState<'all' | 'value' | 'toxic'>('all');

  const teamsWithLeverage = useMemo(() => {
    return MOCK_TEAMS.map(team => {
      const lev = getLeverageScore(team.seed, team.name);
      return { ...team, leverage: lev };
    }).sort((a, b) => b.leverage - a.leverage);
  }, []);

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
                <span style={{ fontSize: '11px', color: '#d4a843', fontWeight: 600 }}>+{(team.adj_oe - team.adj_de).toFixed(1)}</span>
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
