'use client';
import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MOCK_TEAMS, generate2026Brackets, getConferenceName } from '@/lib/mock-data';

const REGIONS = ['East', 'South', 'West', 'Midwest'];

export function Projections2026() {
  const brackets = useMemo(() => generate2026Brackets(), []);
  const [activeTab, setActiveTab] = useState<'overview' | 'analysis'>('analysis');
  const [activeVariant, setActiveVariant] = useState(0);
  const [activeRegion, setActiveRegion] = useState<string | null>(null);
  const variant = brackets[activeVariant];

  const regionTeams = useMemo(() => {
    const map: Record<string, any[]> = {};
    REGIONS.forEach(r => { map[r] = MOCK_TEAMS.filter(t => t.region === r).sort((a, b) => a.seed - b.seed); });
    return map;
  }, []);

  const injuredTeams = useMemo(() => MOCK_TEAMS.filter(t => t.injured.length > 0), []);
  const topCoaches = useMemo(() =>
    [...MOCK_TEAMS]
      .filter(t => {
        const parts = t.coachTourneyRecord.split('-');
        return parts.length === 2 && parseInt(parts[0]) >= 5;
      })
      .sort((a, b) => {
        const aWins = parseInt(a.coachTourneyRecord);
        const bWins = parseInt(b.coachTourneyRecord);
        return bWins - aWins;
      })
      .slice(0, 10),
  []);

  return (
    <div className="p-5 max-w-5xl mx-auto pb-20">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#d4a843" strokeWidth="1.5">
            <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6" />
            <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18" />
            <path d="M4 22h16" />
            <path d="M10 22V9" />
            <path d="M14 22V9" />
            <path d="M6 9V4h12v5a6 6 0 0 1-12 0Z" />
          </svg>
          <div>
            <h1 style={{ fontFamily: 'var(--font-space-grotesk)', color: '#d4a843', fontWeight: 700, fontSize: '20px', letterSpacing: '0.05em' }}>
              2026 PROJECTIONS & INTEL
            </h1>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.04em', marginTop: '2px' }}>
              MARCH MADNESS ORACLE v2.0 • GENERATED MARCH 2026
            </div>
          </div>
        </div>

        {/* Tab Selector */}
        <div className="flex bg-[#1a1208] p-1 rounded-lg border border-[#ff6b35]/20">
          <button
            onClick={() => setActiveTab('analysis')}
            className={`px-4 py-1.5 rounded text-xs font-semibold transition-all ${activeTab === 'analysis' ? 'bg-[#ff6b35]/20 text-[#ff6b35]' : 'text-zinc-500 hover:text-zinc-300'}`}
          >
            DEEP ANALYSIS
          </button>
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-4 py-1.5 rounded text-xs font-semibold transition-all ${activeTab === 'overview' ? 'bg-[#ff6b35]/20 text-[#ff6b35]' : 'text-zinc-500 hover:text-zinc-300'}`}
          >
            FIELD OVERVIEW
          </button>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {activeTab === 'analysis' && (
          <motion.div key="analysis" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-6">
            
            {/* Intel Flags */}
            <div className="glass-wood p-5 border-l-4 border-l-[#d4a843] relative overflow-hidden">
               <div className="absolute top-0 right-0 p-4 opacity-10">
                 <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#d4a843" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
               </div>
               <h2 className="text-sm text-[#d4a843] font-bold tracking-widest mb-4">AUTONOMOUS INTEL FLAGS</h2>
               <div className="space-y-3">
                 <div className="flex gap-3 text-sm">
                   <div className="text-red-500 font-bold mt-0.5">🚨</div>
                   <div><strong className="text-red-400">Cooper Flagg is NOT on Duke's roster.</strong> He entered the 2025 NBA Draft. Evaluate Duke's 2025-26 roster structure carefully before simulating deep runs.</div>
                 </div>
                 <div className="flex gap-3 text-sm">
                   <div className="text-red-500 font-bold mt-0.5">📉</div>
                   <div><strong className="text-orange-400">Iowa State Offensive Crash:</strong> AdjOE dropped to 82nd nationally over last 7 games. Elite defense (13th) but highly vulnerable to variance against 14-seeds.</div>
                 </div>
                 <div className="flex gap-3 text-sm">
                   <div className="text-red-500 font-bold mt-0.5">⚠️</div>
                   <div><strong className="text-orange-400">Vanderbilt Defensive Collapse:</strong> AdjDE fell from 28th → 118th since February. Single largest regression in the field. Prime upset target.</div>
                 </div>
                 <div className="flex gap-3 text-sm">
                   <div className="text-green-500 font-bold mt-0.5">📈</div>
                   <div><strong className="text-green-400">UConn Surging:</strong> Top 5 nationally in AdjDE during current win streak, 11th in AdjOE. Extremely dangerous 2-seed profile.</div>
                 </div>
               </div>
            </div>

            {/* Market Inefficiencies */}
            <div className="glass-panel p-5 border border-red-500/30">
              <h2 className="text-sm font-bold tracking-widest mb-4 text-red-500 flex items-center gap-2">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                FALSE FAVORITES & OVER-SEEDED RISKS
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  { t: 'Purdue', s: '5 (East)', r: 'HIGH', txt: 'Barttorvik luck +0.05. AdjDE regressed heavily in Feb (95.0 from 91.2). Close-game record unsustainable.' },
                  { t: 'Vanderbilt', s: '4 (South)', r: 'EXTREME', txt: 'Defensive metrics collapsed. Offense-only profile in March is highly susceptible to early exit.' },
                  { t: 'Oregon', s: '8 (West)', r: 'MODERATE', txt: 'Turnover rate 21.4% (bottom 30). Vulnerable against pressure defensive schemes.' },
                  { t: 'Mississippi St', s: '8 (East)', r: 'MODERATE', txt: 'Offensive efficiency inflated by weak non-con schedule. Profiles closer to a 10-seed.' }
                ].map(item => (
                  <div key={item.t} className="bg-black/40 p-3 rounded border border-white/5">
                    <div className="flex justify-between items-start mb-2">
                      <div className="font-bold text-zinc-200">{item.t} <span className="text-zinc-500 text-xs ml-1">#{item.s}</span></div>
                      <div className={`text-[10px] px-2 py-0.5 rounded font-bold ${item.r === 'EXTREME' ? 'bg-red-500/20 text-red-400' : item.r === 'HIGH' ? 'bg-orange-500/20 text-orange-400' : 'bg-yellow-500/20 text-yellow-400'}`}>{item.r} RISK</div>
                    </div>
                    <div className="text-xs text-zinc-400 leading-relaxed">{item.txt}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Cinderella Watchlist */}
            <div className="glass-wood p-5 border border-green-500/30">
              <h2 className="text-sm font-bold tracking-widest mb-4 text-green-400 flex items-center gap-2">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 8l4 4-4 4M8 12h8"/></svg>
                CINDERELLA WATCHLIST (HIGH-LEVERAGE UPSETS)
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-white/10 text-[10px] text-zinc-500 tracking-wider">
                      <th className="p-2 pb-3">TEAM</th>
                      <th className="p-2 pb-3">TARGET VICTIM</th>
                      <th className="p-2 pb-3 text-center">UPSET %</th>
                      <th className="p-2 pb-3">WHY THEY'RE DANGEROUS</th>
                    </tr>
                  </thead>
                  <tbody className="text-xs">
                    {[
                      { t: 'Yale', s: 13, v: '(4) Vanderbilt', p: '43%', txt: 'Elite half-court offense, low tempo creates variance. Elite eFG%.' },
                      { t: 'Liberty', s: 12, v: '(5) Purdue / Marquette', p: '38%', txt: 'Top-25 2PT% and 3PT%. Deep rotation, high assist rate.' },
                      { t: 'Miami (OH)', s: 11, v: '(5/6) Any', p: '35%', txt: 'Undefeated in MAC. #1 eFG% nationally. Zero public ownership.' },
                      { t: 'VCU', s: 11, v: '(6) Any', p: '33%', txt: 'Havoc defense generates 18+ TOVs. Extreme momentum.' },
                      { t: 'High Point', s: 14, v: '(3) Iowa St / Nebraska', p: '28%', txt: 'Nation\'s #1 steal rate. 90+ PPG average creates chaos potential.' }
                    ].map((row, i) => (
                      <tr key={row.t} className={`border-b border-white/5 ${i % 2 === 0 ? 'bg-white/[0.02]' : ''}`}>
                        <td className="p-3 font-bold text-zinc-200"><span className="text-zinc-500 mr-2 text-[10px]">{row.s}</span>{row.t}</td>
                        <td className="p-3 text-zinc-400">{row.v}</td>
                        <td className="p-3 text-center font-bold text-green-400">{row.p}</td>
                        <td className="p-3 text-zinc-400 max-w-[200px]">{row.txt}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Matchup Interrogator Deep Dives */}
            <div className="glass-panel p-5 border border-[#ff6b35]/20">
               <h2 className="text-sm text-[#ff6b35] font-bold tracking-widest mb-4">MATCHUP INTERROGATOR DEEP DIVES</h2>
               <div className="space-y-4">
                 
                 <div className="bg-[#1a1208] p-4 rounded-lg border border-white/5">
                   <div className="flex justify-between items-center mb-3">
                     <div className="font-bold text-sm tracking-wide">
                       <span className="text-zinc-500 mr-2">4</span>VIRGINIA <span className="text-zinc-500 mx-2">vs</span> <span className="text-zinc-500 mr-2">13</span>YALE
                     </div>
                     <div className="text-xs font-bold px-2 py-1 bg-zinc-800 rounded text-zinc-300">UVA 57% • YALE 43%</div>
                   </div>
                   <p className="text-xs text-zinc-400 mb-3">
                     <strong className="text-orange-400">Tempo Clash Inverted:</strong> Both teams play at a glacial pace (~61 possessions). At 55 total possessions, every made 3-pointer swings win probability by 5.5%. Yale's elite eFG% (0.55) and ball security are perfectly built for this low-possession environment. 
                   </p>
                   <div className="text-[10px] uppercase text-[#ff6b35] font-bold tracking-widest bg-[#ff6b35]/10 inline-block px-2 py-1 rounded">oracle recommendation: YALE FOR LEVERAGE IN 1000+ ENTRY POOLS</div>
                 </div>

                 <div className="bg-[#1a1208] p-4 rounded-lg border border-white/5">
                   <div className="flex justify-between items-center mb-3">
                     <div className="font-bold text-sm tracking-wide">
                       <span className="text-zinc-500 mr-2">5</span>PURDUE <span className="text-zinc-500 mx-2">vs</span> <span className="text-zinc-500 mr-2">12</span>LIBERTY
                     </div>
                     <div className="text-xs font-bold px-2 py-1 bg-zinc-800 rounded text-zinc-300">PUR 62% • LIB 38%</div>
                   </div>
                   <p className="text-xs text-zinc-400 mb-3">
                     Purdue's unsustainably good close-game luck (+0.05) masks defensive regression. Liberty attacks the rim (high FTA rate) and plays elite defense (91.8 AdjDE). This is a prime 12-over-5 upset candidate based on underlying metrics completely decoupled from public perception.
                   </p>
                   <div className="text-[10px] uppercase text-[#ff6b35] font-bold tracking-widest bg-[#ff6b35]/10 inline-block px-2 py-1 rounded">oracle recommendation: LIBERTY IS HIGHEST-LEVERAGE PLAY IN EAST</div>
                 </div>

               </div>
            </div>

            {/* Optimal Bracket Strategy */}
            <div className="glass-wood p-5 border-t-2 border-[#ff6b35]">
               <h2 className="text-sm font-bold tracking-widest mb-4">MEGA-POOL OPTIMAL PATH (10,000+ ENTRIES)</h2>
               <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-zinc-300">
                 <div className="bg-black/30 p-3 rounded">
                   <div className="text-[10px] text-zinc-500 tracking-widest mb-2 font-bold">R64 DIFFERENTIATORS</div>
                   <ul className="list-disc pl-4 space-y-1 text-green-400">
                     <li>(12) Liberty over (5) Purdue</li>
                     <li>(13) Yale over (4) Virginia</li>
                     <li>(13) Vermont over (4) Vanderbilt</li>
                     <li>(9) TCU over (8) Miss. State</li>
                   </ul>
                 </div>
                 <div className="bg-black/30 p-3 rounded">
                   <div className="text-[10px] text-zinc-500 tracking-widest mb-2 font-bold">DEEP RUN VALUE</div>
                   <ul className="list-disc pl-4 space-y-1">
                     <li>Michigan St clears Illinois path</li>
                     <li>Houston reaches Elite 8</li>
                     <li>Florida outlasts South chaos</li>
                   </ul>
                 </div>
                 <div className="bg-black/30 p-3 rounded">
                   <div className="text-[10px] text-zinc-500 tracking-widest mb-2 font-bold">CHAMPIONSHIP EDGE</div>
                   <div className="text-center mt-2">
                     <div className="text-[#d4a843] font-bold mb-1">Duke 54% vs Michigan 46%</div>
                     <div className="text-[10px] text-zinc-500 leading-tight">Pick Duke in &lt;1000 pools.<br/>Pick Michigan in mega-pools for raw EV leverage.</div>
                   </div>
                 </div>
               </div>
            </div>

          </motion.div>
        )}

        {/* Existing Overview Tab */}
        {activeTab === 'overview' && (
          <motion.div key="overview" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
            {/* === 3 Bracket Variants === */}
            <div className="glass-highlight p-4 rounded-lg mb-5" style={{ borderTop: '2px solid #d4a843' }}>
              <div style={{ fontSize: '10px', color: '#d4a843', letterSpacing: '0.1em', marginBottom: '12px', fontWeight: 700 }}>
                🏆 THREE BRACKET STRATEGIES
              </div>

              {/* Variant selector tabs */}
              <div className="flex gap-2 mb-4">
                {brackets.map((b, i) => (
                  <motion.button
                    key={b.name}
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={() => setActiveVariant(i)}
                    className="flex-1 px-3 py-3 rounded-lg text-center"
                    style={{
                      background: activeVariant === i ? 'rgba(255,107,53,0.15)' : 'rgba(255,107,53,0.04)',
                      border: `1px solid ${activeVariant === i ? '#ff6b35' : 'rgba(255,107,53,0.12)'}`,
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ fontSize: '16px', marginBottom: '4px' }}>{b.icon}</div>
                    <div style={{ fontSize: '12px', fontWeight: 700, color: activeVariant === i ? '#ff6b35' : 'var(--text-secondary)' }}>{b.name}</div>
                    <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px' }}>{b.description}</div>
                  </motion.button>
                ))}
              </div>

              {/* Active variant detail */}
              <AnimatePresence mode="wait">
                <motion.div key={variant.name} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '12px' }}>
                    {variant.finalFour.map((team, i) => {
                      const data = MOCK_TEAMS.find(t => t.name === team);
                      return (
                        <div key={team} className="glass-wood p-3 rounded-lg text-center" style={{ borderTop: i === 0 ? '2px solid #d4a843' : '1px solid rgba(255,107,53,0.12)' }}>
                          <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>#{data?.seed} SEED • {data ? getConferenceName(data.conference) : ''}</div>
                          <div style={{ fontSize: '14px', fontWeight: 700, color: '#ff6b35', margin: '4px 0' }}>{team}</div>
                          <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>{data?.record}</div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Championship Game */}
                  <div className="glass-wood p-3 rounded-lg text-center mb-3" style={{ border: '1px solid rgba(212,168,67,0.3)' }}>
                    <div style={{ fontSize: '9px', color: '#d4a843', letterSpacing: '0.08em', fontWeight: 600 }}>CHAMPIONSHIP PREDICTION</div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', marginTop: '8px' }}>
                      <span style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>{variant.championshipGame[0]}</span>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>vs</span>
                      <span style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>{variant.championshipGame[1]}</span>
                    </div>
                    <div style={{ marginTop: '8px' }}>
                      <span style={{ fontSize: '9px', color: '#d4a843', letterSpacing: '0.08em' }}>CHAMPION: </span>
                      <span style={{ fontSize: '14px', fontWeight: 800, color: '#d4a843' }}>{variant.champion} 🏆</span>
                    </div>
                  </div>

                  {/* Predicted Upsets */}
                  {variant.upsets.length > 0 && (
                    <div className="glass-wood p-3 rounded-lg" style={{ borderLeft: '3px solid #e74c3c' }}>
                      <div style={{ fontSize: '9px', color: '#e74c3c', letterSpacing: '0.08em', fontWeight: 600, marginBottom: '6px' }}>⚡ PREDICTED UPSETS</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '4px' }}>
                        {variant.upsets.map((u, i) => (
                          <div key={i} style={{ fontSize: '11px', color: 'var(--text-secondary)', padding: '3px 0' }}>
                            <span style={{ color: '#e74c3c', fontWeight: 600 }}>({u.winnerSeed})</span> {u.winner} over <span style={{ color: 'var(--text-muted)' }}>({u.loserSeed})</span> {u.loser} • {u.round}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            </div>

            {/* === Projected 68-Team Field === */}
            <div className="glass-wood p-4 rounded-lg mb-5">
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.05em', marginBottom: '12px', fontWeight: 600 }}>PROJECTED 68-TEAM FIELD</div>
              <div className="flex gap-2 mb-3">
                <button
                  onClick={() => setActiveRegion(null)}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ background: !activeRegion ? 'rgba(255,107,53,0.15)' : 'transparent', border: `1px solid ${!activeRegion ? '#ff6b35' : 'rgba(255,107,53,0.15)'}`, color: !activeRegion ? '#ff6b35' : 'var(--text-muted)', cursor: 'pointer' }}
                >All</button>
                {REGIONS.map(r => (
                  <button key={r} onClick={() => setActiveRegion(r)} className="px-3 py-1.5 rounded text-xs"
                    style={{ background: activeRegion === r ? 'rgba(255,107,53,0.15)' : 'transparent', border: `1px solid ${activeRegion === r ? '#ff6b35' : 'rgba(255,107,53,0.15)'}`, color: activeRegion === r ? '#ff6b35' : 'var(--text-muted)', cursor: 'pointer' }}
                  >{r}</button>
                ))}
              </div>

              {(activeRegion ? [activeRegion] : REGIONS).map(region => (
                <div key={region} className="mb-4">
                  <div style={{ fontSize: '11px', color: '#ff6b35', fontWeight: 600, marginBottom: '6px', letterSpacing: '0.04em' }}>{region.toUpperCase()} REGION</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '30px 1fr 50px 80px 60px 60px 60px', gap: '4px', alignItems: 'center' }}>
                    <span style={{ fontSize: '8px', color: 'var(--text-muted)' }}>SEED</span><span style={{ fontSize: '8px', color: 'var(--text-muted)' }}>TEAM</span><span style={{ fontSize: '8px', color: 'var(--text-muted)' }}>REC</span><span style={{ fontSize: '8px', color: 'var(--text-muted)' }}>CONF</span><span style={{ fontSize: '8px', color: 'var(--text-muted)', textAlign: 'center' }}>OE</span><span style={{ fontSize: '8px', color: 'var(--text-muted)', textAlign: 'center' }}>DE</span><span style={{ fontSize: '8px', color: 'var(--text-muted)', textAlign: 'right' }}>NET</span>
                    {regionTeams[region]?.map((team, i) => (
                      <motion.div key={team.name} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.02 }} style={{ display: 'contents' }}>
                        <span style={{ fontSize: '11px', fontWeight: 700, color: team.seed <= 4 ? '#ff6b35' : 'var(--text-muted)', width: '22px', height: '22px', borderRadius: '50%', background: 'rgba(255,107,53,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{team.seed}</span>
                        <span style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-primary)' }}>{team.name}{team.injured.length > 0 && <span style={{ color: '#e74c3c', fontSize: '9px', marginLeft: '4px' }}>⚠</span>}</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{team.record}</span>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{getConferenceName(team.conference)}</span>
                        <span style={{ fontSize: '11px', color: '#2ecc71', textAlign: 'center' }}>{team.adj_oe.toFixed(1)}</span>
                        <span style={{ fontSize: '11px', color: '#3498db', textAlign: 'center' }}>{team.adj_de.toFixed(1)}</span>
                        <span style={{ fontSize: '11px', fontWeight: 600, textAlign: 'right', color: (team.adj_oe - team.adj_de) > 25 ? '#2ecc71' : (team.adj_oe - team.adj_de) > 20 ? '#d4a843' : 'var(--text-secondary)' }}>+{(team.adj_oe - team.adj_de).toFixed(1)}</span>
                      </motion.div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Content Bottom Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="glass-wood p-4 rounded-lg">
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.05em', marginBottom: '10px', fontWeight: 600 }}>🎓 TOP MARCH COACHES</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: '6px 12px' }}>
                  {topCoaches.slice(0, 5).map((t, i) => (
                    <motion.div key={t.name} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }} style={{ display: 'contents' }}>
                      <div><span style={{ fontSize: '12px', fontWeight: 600, color: '#ff6b35' }}>{t.coachName}</span><span style={{ fontSize: '10px', color: 'var(--text-muted)', marginLeft: '6px' }}>({t.name})</span></div>
                      <span style={{ fontSize: '12px', fontWeight: 600, color: '#2ecc71' }}>{t.coachTourneyRecord}</span>
                      <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{(() => { const p = t.coachTourneyRecord.split('-'); return p.length === 2 ? `${((parseInt(p[0]) / (parseInt(p[0]) + parseInt(p[1]))) * 100).toFixed(0)}%` : ''; })()}</span>
                    </motion.div>
                  ))}
                </div>
              </div>

              {injuredTeams.length > 0 && (
                <div className="glass-wood p-4 rounded-lg" style={{ borderLeft: '3px solid #e74c3c' }}>
                  <div style={{ fontSize: '10px', color: '#e74c3c', letterSpacing: '0.05em', marginBottom: '8px', fontWeight: 600 }}>🏥 KEY INJURIES</div>
                  {injuredTeams.map(t => (
                    <div key={t.name} style={{ marginBottom: '6px' }}>
                      <span style={{ fontSize: '12px', fontWeight: 600, color: '#ff6b35' }}>{t.name}</span>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>{t.injured.join(', ')} — {t.seed <= 4 ? 'Seeding risk' : 'Monitor availability'}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
