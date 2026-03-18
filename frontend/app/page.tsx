'use client';
import { useState, useEffect } from 'react';
import { Sidebar } from '@/components/nav/Sidebar';
import { PageTransition } from '@/components/nav/PageTransition';
import { ModelStatusBadge } from '@/components/nav/ModelStatusBadge';
import { PowerRankings } from '@/components/rankings/PowerRankings';
import { MatchupAnalyzer } from '@/components/matchup/MatchupOracle';
import { Projections2026 } from '@/components/projections/Projections2026';
import { WarRoomMatrix } from '@/components/warroom/WarRoomMatrix';
import { BasketballCourt } from '@/components/ui/BasketballCourt';
import type { NavPage } from '@/components/nav/Sidebar';
import dynamic from 'next/dynamic';
const FullBracket = dynamic(
  () => import('@/components/bracket/BracketSimulator').then(m => ({ default: m.FullBracket })),
  { ssr: false, loading: () => <div style={{ color: '#ff6b35', padding: '2rem', textAlign: 'center' }}>Loading bracket...</div> }
);

export default function HomePage() {
  const [activePage, setActivePage] = useState<NavPage>('projections');
  const [season, setSeason] = useState(2026);
  const [modelStatus, setModelStatus] = useState<'online' | 'offline' | 'loading'>('loading');

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then((r) => r.ok ? setModelStatus('online') : setModelStatus('offline'))
      .catch(() => setModelStatus('offline'));
  }, []);

  return (
    <div style={{ display: 'flex', height: '100vh', background: 'var(--hardwood-dark)', overflow: 'hidden', position: 'relative' }}>
      {/* Court watermark */}
      <div style={{
        position: 'fixed', top: '50%', left: '55%',
        transform: 'translate(-50%, -50%) rotate(90deg)',
        width: '80vh', height: '80vh',
        pointerEvents: 'none', zIndex: 0,
      }}>
        <BasketballCourt opacity={0.03} />
      </div>

      <Sidebar activePage={activePage} onNavigate={setActivePage} modelStatus={modelStatus} season={season} onSeasonChange={setSeason} />

      <main style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', position: 'relative', zIndex: 1 }}>
        {/* Top bar */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 20px',
          borderBottom: '1px solid rgba(255,107,53,0.08)',
          background: 'rgba(26, 18, 8, 0.9)',
          backdropFilter: 'blur(12px)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ff6b35" strokeWidth="1.5">
              <circle cx="12" cy="12" r="10" />
              <path d="M 2 12 H 22" />
              <path d="M 12 2 C 16 6, 16 18, 12 22" />
              <path d="M 12 2 C 8 6, 8 18, 12 22" />
            </svg>
            <span style={{ fontFamily: 'var(--font-space-grotesk)', color: '#ff6b35', fontWeight: 700, letterSpacing: '0.1em', fontSize: '14px' }}>
              MADNESS MATRIX
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <ModelStatusBadge status={modelStatus} version="v2.0" />
          </div>
        </div>

        <div style={{ flex: 1, overflow: 'auto' }}>
          <PageTransition pageKey={activePage}>
            {activePage === 'rankings' && <PowerRankings />}
            {activePage === 'matchup' && <MatchupAnalyzer />}
            {activePage === 'bracket' && <FullBracket />}
            {activePage === 'projections' && <Projections2026 />}
            {activePage === 'warroom' && <WarRoomMatrix />}
          </PageTransition>
        </div>
      </main>
    </div>
  );
}
