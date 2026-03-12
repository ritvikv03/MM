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
import type { GraphResponse, TeamNode } from '@/lib/api-types';
import { fetchGraph } from '@/lib/api';
import { getMockGraph } from '@/lib/mock-data';

function StubDataBanner({ dataSource }: { dataSource: 'real' | 'stub' | 'unknown' }) {
  if (dataSource !== 'stub') return null;
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      zIndex: 9999,
      background: '#d4a843',
      color: '#1a1208',
      textAlign: 'center',
      padding: '6px 16px',
      fontSize: '12px',
      fontWeight: 700,
      letterSpacing: '0.04em',
      lineHeight: 1.4,
    }}>
      ⚠ Synthetic data active — real model offline. Results are illustrative only.
    </div>
  );
}

import dynamic from 'next/dynamic';
const FullBracket = dynamic(
  () => import('@/components/bracket/BracketSimulator').then(m => ({ default: m.FullBracket })),
  { ssr: false, loading: () => <div style={{ color: '#ff6b35', padding: '2rem', textAlign: 'center' }}>Loading bracket...</div> }
);
const ConstellationCanvas = dynamic(
  () => import('@/components/constellation/ConstellationCanvas').then(m => ({ default: m.ConstellationCanvas })),
  { ssr: false, loading: () => <div style={{ color: '#00f5ff', padding: '2rem', textAlign: 'center' }}>Loading graph...</div> }
);

export default function HomePage() {
  const [activePage, setActivePage] = useState<NavPage>('projections');
  const [season, setSeason] = useState(2026);
  const [modelStatus, setModelStatus] = useState<'online' | 'offline' | 'loading'>('loading');
  const [graphData, setGraphData] = useState<GraphResponse>(getMockGraph());
  const [graphLoading, setGraphLoading] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);
  const [dataSource, setDataSource] = useState<'real' | 'stub' | 'unknown'>('unknown');

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then((r) => r.ok ? setModelStatus('online') : setModelStatus('offline'))
      .catch(() => setModelStatus('offline'));
  }, []);

  // Reload graph when season changes
  useEffect(() => {
    setGraphLoading(true);
    setDataSource('unknown');
    fetchGraph(season)
      .then((data) => {
        setGraphData(data);
        setDataSource(data.data_source === 'real' ? 'real' : 'stub');
      })
      .finally(() => setGraphLoading(false));
  }, [season]);

  const handleSelectTeam = (team: TeamNode) => {
    setSelectedTeam(team.id);
  };

  return (
    <div style={{ display: 'flex', height: '100vh', background: 'var(--hardwood-dark)', overflow: 'hidden', position: 'relative' }}>
      <StubDataBanner dataSource={dataSource} />
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
              MARCH MADNESS ORACLE
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {modelStatus === 'offline' && (
              <span style={{ fontSize: '9px', color: '#d4a843', letterSpacing: '0.05em', fontWeight: 600 }}>DEMO MODE</span>
            )}
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
            {activePage === 'graph' && (
              <div style={{ height: 'calc(100vh - 56px)', position: 'relative' }}>
                {graphLoading && (
                  <div style={{
                    position: 'absolute', top: 12, right: 12, zIndex: 10,
                    fontSize: '9px', color: '#00f5ff', background: 'rgba(0,245,255,0.1)',
                    padding: '4px 8px', borderRadius: '4px', letterSpacing: '0.06em',
                  }}>
                    LOADING SEASON {season}…
                  </div>
                )}
                <ConstellationCanvas
                  graph={graphData}
                  selectedTeam={selectedTeam}
                  onSelectTeam={handleSelectTeam}
                />
              </div>
            )}
          </PageTransition>
        </div>
      </main>
    </div>
  );
}
