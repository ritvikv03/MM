'use client';
import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { fetchMatchup } from '@/lib/api';
import { getConferenceName, TOURNAMENT_TEAMS_2026 as MOCK_TEAMS } from '@/lib/team-data';
import type { EnrichedMatchupResponse } from '@/lib/api-types';

const TEAMS = MOCK_TEAMS.map(t => t.name);

export function MatchupAnalyzer() {
  const [homeTeam, setHomeTeam] = useState<string | null>(null);
  const [awayTeam, setAwayTeam] = useState<string | null>(null);
  const [result, setResult] = useState<EnrichedMatchupResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectingFor, setSelectingFor] = useState<'home' | 'away' | null>(null);

  const runAnalysis = useCallback(async () => {
    if (!homeTeam || !awayTeam) return;
    setLoading(true);
    try {
      const res = await fetchMatchup({ home_team: homeTeam, away_team: awayTeam, season: 2026, neutral_site: true });
      setResult(res);
    } catch { /* handled by api layer */ }
    finally { setLoading(false); }
  }, [homeTeam, awayTeam]);

  const isFavorite = (team: 'home' | 'away') => {
    if (!result) return false;
    return team === 'home' ? result.p_win_home >= 0.5 : result.p_win_home < 0.5;
  };

  return (
    <div className="p-5 max-w-3xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ff6b35" strokeWidth="1.5">
          <circle cx="12" cy="12" r="10" />
          <path d="M 2 12 H 22" />
          <path d="M 12 2 C 16 6, 16 18, 12 22" />
          <path d="M 12 2 C 8 6, 8 18, 12 22" />
        </svg>
        <h1 style={{ fontFamily: 'var(--font-space-grotesk)', color: '#ff6b35', fontWeight: 700, fontSize: '20px', letterSpacing: '0.05em' }}>
          MATCHUP ANALYZER
        </h1>
      </div>

      {/* Team Selectors */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '12px', alignItems: 'center', marginBottom: '16px' }}>
        <TeamSelector label="TEAM A" team={homeTeam} onClick={() => setSelectingFor(selectingFor === 'home' ? null : 'home')} isActive={selectingFor === 'home'} />
        <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-muted)' }}>VS</span>
        <TeamSelector label="TEAM B" team={awayTeam} onClick={() => setSelectingFor(selectingFor === 'away' ? null : 'away')} isActive={selectingFor === 'away'} />
      </div>

      {/* Team Picker Grid */}
      <AnimatePresence>
        {selectingFor && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden mb-4">
            <div className="glass-wood p-3 rounded-lg" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
              {TEAMS.map((t) => (
                <button
                  key={t}
                  onClick={() => {
                    if (selectingFor === 'home') setHomeTeam(t);
                    else setAwayTeam(t);
                    setSelectingFor(null);
                    setResult(null);
                  }}
                  disabled={(selectingFor === 'home' && t === awayTeam) || (selectingFor === 'away' && t === homeTeam)}
                  className="px-2 py-2 rounded-lg text-xs transition-all"
                  style={{
                    background: 'rgba(255,107,53,0.06)',
                    border: '1px solid rgba(255,107,53,0.15)',
                    color: '#f5f0e8',
                    cursor: 'pointer',
                    opacity: (selectingFor === 'home' && t === awayTeam) || (selectingFor === 'away' && t === homeTeam) ? 0.3 : 1,
                  }}
                >{t}</button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Analyze Button */}
      <div className="flex justify-center mb-5">
        <motion.button
          whileHover={loading ? {} : { scale: 1.04 }}
          whileTap={loading ? {} : { scale: 0.97 }}
          disabled={!homeTeam || !awayTeam || homeTeam === awayTeam || loading}
          onClick={runAnalysis}
          className="px-8 py-3 rounded-lg font-semibold text-sm"
          style={{
            background: 'rgba(255, 107, 53, 0.15)',
            border: '1px solid #ff6b35',
            color: '#ff6b35',
            opacity: (!homeTeam || !awayTeam || loading) ? 0.5 : 1,
            cursor: (!homeTeam || !awayTeam || loading) ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Analyzing...' : '🏀 Analyze Matchup'}
        </motion.button>
      </div>

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex flex-col gap-4">
            {/* Odds Display */}
            <div className="glass-highlight p-5 rounded-lg">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '16px', alignItems: 'center', textAlign: 'center' }}>
                <div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', letterSpacing: '0.05em' }}>{result.home_team}</div>
                  <div style={{ fontSize: '32px', fontWeight: 800, color: isFavorite('home') ? '#2ecc71' : '#e74c3c' }}>
                    {(result.p_win_home * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>ML {result.home_moneyline}</div>
                </div>

                <div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>SPREAD</div>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: '#d4a843' }}>
                    {result.spread_mean > 0 ? '+' : ''}{result.spread_mean.toFixed(1)}
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', letterSpacing: '0.05em' }}>{result.away_team}</div>
                  <div style={{ fontSize: '32px', fontWeight: 800, color: isFavorite('away') ? '#2ecc71' : '#e74c3c' }}>
                    {((1 - result.p_win_home) * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>ML {result.away_moneyline}</div>
                </div>
              </div>
            </div>

            {/* Upset Probability Gauge */}
            <div className="glass-wood p-4 rounded-lg">
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.05em', marginBottom: '8px' }}>UPSET PROBABILITY</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div className="stat-bar" style={{ flex: 1, height: '12px' }}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.upset_probability * 100}%` }}
                    transition={{ duration: 0.8 }}
                    className="stat-bar-fill"
                    style={{
                      background: result.upset_probability > 0.4 ? 'linear-gradient(90deg, #d4a843, #e74c3c)' : 'linear-gradient(90deg, rgba(46,204,113,0.5), #2ecc71)',
                    }}
                  />
                </div>
                <span style={{ fontSize: '16px', fontWeight: 700, color: result.upset_probability > 0.4 ? '#e74c3c' : '#2ecc71', minWidth: '50px', textAlign: 'right' }}>
                  {(result.upset_probability * 100).toFixed(0)}%
                </span>
              </div>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '6px' }}>
                {result.upset_probability > 0.4
                  ? `⚠ Historically, ${Math.max(result.home_seed, result.away_seed)}-seeds beat ${Math.min(result.home_seed, result.away_seed)}-seeds ${(result.upset_probability * 100).toFixed(0)}% of the time`
                  : `Favored team expected to win. Upset is unlikely in this matchup.`
                }
              </div>
            </div>

            {/* Head-to-Head Stats */}
            <div className="glass-wood p-4 rounded-lg">
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.05em', marginBottom: '12px' }}>HEAD-TO-HEAD COMPARISON</div>
              <StatCompareRow label="Seed" home={`#${result.home_seed}`} away={`#${result.away_seed}`} homeWins={result.home_seed < result.away_seed} />
              <StatCompareRow label="Record" home={result.home_record} away={result.away_record} homeWins={false} />
              <StatCompareRow label="Adj OE" home={result.home_adj_oe.toFixed(1)} away={result.away_adj_oe.toFixed(1)} homeWins={result.home_adj_oe > result.away_adj_oe} />
              <StatCompareRow label="Adj DE" home={result.home_adj_de.toFixed(1)} away={result.away_adj_de.toFixed(1)} homeWins={result.home_adj_de < result.away_adj_de} />
              <StatCompareRow label="Tempo" home={result.home_tempo.toFixed(1)} away={result.away_tempo.toFixed(1)} homeWins={false} />
              <StatCompareRow label="Key Player" home={result.home_key_player} away={result.away_key_player} homeWins={false} />
            </div>

            {/* Why They Win Factors */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div className="glass-wood p-4 rounded-lg" style={{ borderTop: '2px solid #2ecc71' }}>
                <div style={{ fontSize: '10px', color: '#2ecc71', letterSpacing: '0.05em', marginBottom: '8px', fontWeight: 600 }}>
                  🏆 WHY {result.home_team.toUpperCase()} WINS
                </div>
                {result.home_factors.map((f, i) => (
                  <div key={i} style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '6px', paddingLeft: '8px', borderLeft: '2px solid rgba(46,204,113,0.3)' }}>
                    {f}
                  </div>
                ))}
              </div>
              <div className="glass-wood p-4 rounded-lg" style={{ borderTop: '2px solid #3498db' }}>
                <div style={{ fontSize: '10px', color: '#3498db', letterSpacing: '0.05em', marginBottom: '8px', fontWeight: 600 }}>
                  🏆 WHY {result.away_team.toUpperCase()} WINS
                </div>
                {result.away_factors.map((f, i) => (
                  <div key={i} style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '6px', paddingLeft: '8px', borderLeft: '2px solid rgba(52,152,219,0.3)' }}>
                    {f}
                  </div>
                ))}
              </div>
            </div>

            {/* Luck Warning */}
            {result.luck_compressed && (
              <div className="glass-wood p-3 rounded-lg" style={{ borderLeft: '3px solid #d4a843' }}>
                <span style={{ fontSize: '11px', color: '#d4a843' }}>
                  ⚠ Clutch Regression Warning: One of these teams has an unsustainable close-game record. The Bayesian model has regressed their clutch metrics toward the mean.
                </span>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TeamSelector({ label, team, onClick, isActive }: { label: string; team: string | null; onClick: () => void; isActive: boolean }) {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      onClick={onClick}
      className="glass-wood p-4 rounded-lg text-center w-full"
      style={{ border: isActive ? '1px solid #ff6b35' : '1px solid rgba(255,107,53,0.12)', cursor: 'pointer' }}
    >
      <div style={{ fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.1em', marginBottom: '6px' }}>{label}</div>
      {team ? (
        <div style={{ fontSize: '15px', fontWeight: 700, color: '#ff6b35' }}>{team}</div>
      ) : (
        <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Select Team</div>
      )}
    </motion.button>
  );
}

function StatCompareRow({ label, home, away, homeWins }: { label: string; home: string; away: string; homeWins: boolean }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '8px', alignItems: 'center', padding: '5px 0', borderBottom: '1px solid rgba(255,107,53,0.06)' }}>
      <span style={{ fontSize: '12px', fontWeight: homeWins ? 700 : 400, color: homeWins ? '#2ecc71' : 'var(--text-secondary)', textAlign: 'right' }}>
        {home}
      </span>
      <span style={{ fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.05em', minWidth: '60px', textAlign: 'center' }}>{label}</span>
      <span style={{ fontSize: '12px', fontWeight: !homeWins && label !== 'Record' && label !== 'Tempo' && label !== 'Key Player' ? 700 : 400, color: !homeWins && label !== 'Record' && label !== 'Tempo' && label !== 'Key Player' ? '#2ecc71' : 'var(--text-secondary)' }}>
        {away}
      </span>
    </div>
  );
}
