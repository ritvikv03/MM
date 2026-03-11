'use client';
import { motion } from 'framer-motion';

export type NavPage = 'rankings' | 'matchup' | 'bracket' | 'projections' | 'warroom' | 'graph';

interface SidebarProps {
  activePage: NavPage;
  onNavigate: (page: NavPage) => void;
  modelStatus: 'online' | 'offline' | 'loading';
  season: number;
  onSeasonChange: (season: number) => void;
}

function RankingsIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round">
      <rect x="3" y="14" width="4" height="7" rx="1" />
      <rect x="10" y="8" width="4" height="13" rx="1" />
      <rect x="17" y="3" width="4" height="18" rx="1" />
    </svg>
  );
}

function MatchupIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M 2 12 H 22" />
      <path d="M 12 2 C 16 6, 16 18, 12 22" />
      <path d="M 12 2 C 8 6, 8 18, 12 22" />
    </svg>
  );
}

function BracketIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round">
      <path d="M 4 4 L 10 4 L 10 10" />
      <path d="M 4 20 L 10 20 L 10 14" />
      <path d="M 10 7 L 16 7 L 16 12" />
      <path d="M 10 17 L 16 17 L 16 12" />
      <path d="M 16 12 L 20 12" />
      <circle cx="20" cy="12" r="1.5" fill={color} />
    </svg>
  );
}

function TrophyIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round">
      <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6" />
      <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18" />
      <path d="M4 22h16" />
      <path d="M10 22V9" />
      <path d="M14 22V9" />
      <path d="M6 9V4h12v5a6 6 0 0 1-12 0Z" />
    </svg>
  );
}

function EyeIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function GraphIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round">
      <circle cx="5" cy="5" r="2" />
      <circle cx="19" cy="5" r="2" />
      <circle cx="12" cy="19" r="2" />
      <circle cx="5" cy="12" r="1.5" />
      <line x1="5" y1="7" x2="5" y2="10.5" />
      <line x1="7" y1="5" x2="17" y2="5" />
      <line x1="19" y1="7" x2="13.5" y2="17.5" />
      <line x1="5" y1="13.5" x2="10.5" y2="17.5" />
    </svg>
  );
}

const NAV_ITEMS: { id: NavPage; label: string; Icon: React.FC<{ color: string }> }[] = [
  { id: 'rankings', label: 'Rankings', Icon: RankingsIcon },
  { id: 'matchup', label: 'Matchup', Icon: MatchupIcon },
  { id: 'bracket', label: 'Bracket', Icon: BracketIcon },
  { id: 'projections', label: '2026', Icon: TrophyIcon },
  { id: 'warroom', label: 'War Room', Icon: EyeIcon },
  { id: 'graph', label: 'Graph', Icon: GraphIcon },
];

export function Sidebar({ activePage, onNavigate, modelStatus, season, onSeasonChange }: SidebarProps) {
  const SEASONS = [2022, 2023, 2024, 2025];
  const statusColors = { online: '#2ecc71', offline: '#e74c3c', loading: '#d4a843' };

  return (
    <motion.nav
      initial={{ x: -20, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="flex flex-col h-full py-6 px-3 gap-2"
      style={{
        width: '72px',
        background: 'rgba(26, 18, 8, 0.9)',
        backdropFilter: 'blur(20px)',
        borderRight: '1px solid rgba(255, 107, 53, 0.12)',
      }}
    >
      {/* Logo */}
      <div className="flex justify-center mb-4">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#ff6b35" strokeWidth="1.8">
          <circle cx="12" cy="12" r="10" />
          <path d="M 2 12 H 22" />
          <path d="M 12 2 C 16 6, 16 18, 12 22" />
          <path d="M 12 2 C 8 6, 8 18, 12 22" />
        </svg>
      </div>

      {NAV_ITEMS.map((item) => {
        const isActive = activePage === item.id;
        const color = isActive ? '#ff6b35' : 'rgba(245, 240, 232, 0.35)';
        return (
          <motion.button
            key={item.id}
            onClick={() => onNavigate(item.id)}
            title={item.label}
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.95 }}
            className="flex flex-col items-center gap-1 py-2.5 px-1 rounded-lg w-full"
            style={{
              background: isActive ? 'rgba(255, 107, 53, 0.12)' : 'transparent',
              border: `1px solid ${isActive ? 'rgba(255, 107, 53, 0.3)' : 'transparent'}`,
            }}
          >
            <item.Icon color={color} />
            <span style={{ fontSize: '8px', letterSpacing: '0.06em', color, fontWeight: isActive ? 600 : 400 }}>
              {item.label.toUpperCase()}
            </span>
          </motion.button>
        );
      })}

      <div className="flex-1" />

      <div className="flex flex-col items-center gap-1 mb-2">
        <span style={{ fontSize: '8px', color: 'var(--text-muted)', letterSpacing: '0.05em' }}>SEASON</span>
        <select
          value={season}
          onChange={(e) => onSeasonChange(Number(e.target.value))}
          style={{
            background: 'rgba(255, 107, 53, 0.08)',
            border: '1px solid rgba(255, 107, 53, 0.2)',
            color: '#ff6b35',
            borderRadius: '6px',
            fontSize: '10px',
            padding: '2px 4px',
            width: '56px',
            cursor: 'pointer',
          }}
        >
          {SEASONS.map((y) => (
            <option key={y} value={y} style={{ background: '#1a1208' }}>{y}</option>
          ))}
        </select>
      </div>

      <div className="flex flex-col items-center gap-1">
        <motion.div
          animate={{ opacity: [1, 0.4, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
          style={{
            width: '8px', height: '8px', borderRadius: '50%',
            background: statusColors[modelStatus],
            boxShadow: `0 0 6px ${statusColors[modelStatus]}`,
          }}
        />
        <span style={{ fontSize: '7px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
          {modelStatus}
        </span>
      </div>
    </motion.nav>
  );
}
