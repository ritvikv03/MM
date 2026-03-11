'use client';
import { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MOCK_TEAMS, getConferenceName } from '@/lib/mock-data';
import { simulateBracket } from '@/lib/api';
import { BracketHeatmap } from './BracketHeatmap';
import type { MockTeam } from '@/lib/mock-data';
import type { SimulateResponse } from '@/lib/api-types';

// ─── Constants ───────────────────────────────────────────────────────────────
const REGIONS = ['East', 'South', 'West', 'Midwest'] as const;

function parseCoachWins(record: string): number {
  return parseInt(record.split('-')[0], 10) || 0;
}
type Region = typeof REGIONS[number];
// R64 pairing order: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
const SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15];
type RoundName = 'R64' | 'R32' | 'S16' | 'E8' | 'F4' | 'Championship';

// ─── Types ────────────────────────────────────────────────────────────────────
interface BracketGame {
  id: string;
  round: RoundName;
  region: string;
  pos: number;
  teamA: MockTeam | null;
  teamB: MockTeam | null;
  winner: MockTeam | null;
  winProb: number;     // P(teamA wins)
  isUserPick: boolean;
}

interface InterrogatorData {
  teamA: MockTeam;
  teamB: MockTeam;
  pWin: number;
  narrative: string;
  recommendation: string;
  styleClash: string;
  radarA: number[];
  radarB: number[];
}

// ─── Core model ───────────────────────────────────────────────────────────────
function computeWinProb(a: MockTeam, b: MockTeam, chaos: number, wpaMap: Record<string, number> = {}): number {
  const wpaA = wpaMap[a.name] ?? 0;
  const wpaB = wpaMap[b.name] ?? 0;
  const emA = a.adj_oe - a.adj_de - a.luck * 2.5 + wpaA;
  const emB = b.adj_oe - b.adj_de - b.luck * 2.5 + wpaB;
  const pEff = 1 / (1 + Math.exp(-(emA - emB) / 10));

  const dnaA = Math.min(1, a.sos / 10) * 0.3 + (1 - Math.abs(a.luck) * 5) * 0.3 + Math.min(1, parseCoachWins(a.coachTourneyRecord) / 20) * 0.4;
  const dnaB = Math.min(1, b.sos / 10) * 0.3 + (1 - Math.abs(b.luck) * 5) * 0.3 + Math.min(1, parseCoachWins(b.coachTourneyRecord) / 20) * 0.4;
  const pDna = 1 / (1 + Math.exp(-(dnaA - dnaB) * 5));

  const wEff = 0.7 * (1.2 - chaos * 0.6);
  const wDna = 0.3 * (0.7 + chaos * 0.8);
  const wTotal = wEff + wDna;
  let p = (wEff / wTotal) * pEff + (wDna / wTotal) * pDna;

  if (chaos > 0.6) {
    p = p * (1 - (chaos - 0.6) * 0.5) + 0.5 * (chaos - 0.6) * 0.5;
  }
  return Math.max(0.02, Math.min(0.98, p));
}

function estimateSpread(a: MockTeam, b: MockTeam): number {
  const emDiff = (a.adj_oe - a.adj_de) - (b.adj_oe - b.adj_de);
  return Math.round(emDiff * 0.8 * 2) / 2; // round to nearest 0.5
}

// ─── Bracket computation (pure function) ─────────────────────────────────────
function computeBracket(
  bracketByRegion: Record<string, MockTeam[]>,
  userPicks: Map<string, MockTeam>,
  chaos: number,
  wpaMap: Record<string, number>,
): BracketGame[] {
  const mk = (id: string, round: RoundName, region: string, pos: number, teamA: MockTeam | null, teamB: MockTeam | null): BracketGame => ({
    id, round, region, pos, teamA, teamB, winner: null, winProb: 0.5, isUserPick: false,
  });

  const gm = new Map<string, BracketGame>();

  const resolveGame = (id: string) => {
    const g = gm.get(id)!;
    if (!g.teamA || !g.teamB) return;
    g.winProb = computeWinProb(g.teamA, g.teamB, chaos, wpaMap);
    const modelWinner = g.winProb >= 0.5 ? g.teamA : g.teamB;
    const userPick = userPicks.get(id);
    g.winner = userPick ?? modelWinner;
    g.isUserPick = !!userPick;
  };

  const setSlot = (id: string, slot: 'teamA' | 'teamB', team: MockTeam | null) => {
    const g = gm.get(id);
    if (g && team) g[slot] = team;
  };

  // Seed R64 and create empty downstream games
  REGIONS.forEach(region => {
    const teams = bracketByRegion[region] ?? [];
    for (let i = 0; i < 8; i++) {
      const id = `${region}-R64-${i}`;
      gm.set(id, mk(id, 'R64', region, i, teams[i * 2] ?? null, teams[i * 2 + 1] ?? null));
      resolveGame(id);
    }
    for (let i = 0; i < 4; i++) { const id = `${region}-R32-${i}`; gm.set(id, mk(id, 'R32', region, i, null, null)); }
    for (let i = 0; i < 2; i++) { const id = `${region}-S16-${i}`; gm.set(id, mk(id, 'S16', region, i, null, null)); }
    gm.set(`${region}-E8-0`, mk(`${region}-E8-0`, 'E8', region, 0, null, null));
  });
  gm.set('F4-0', mk('F4-0', 'F4', 'FinalFour', 0, null, null));
  gm.set('F4-1', mk('F4-1', 'F4', 'FinalFour', 1, null, null));
  gm.set('Championship-0', mk('Championship-0', 'Championship', 'Championship', 0, null, null));

  // Cascade within each region: R64→R32→S16→E8
  REGIONS.forEach(region => {
    for (let i = 0; i < 8; i++) {
      const g = gm.get(`${region}-R64-${i}`)!;
      setSlot(`${region}-R32-${Math.floor(i / 2)}`, i % 2 === 0 ? 'teamA' : 'teamB', g.winner);
    }
    for (let i = 0; i < 4; i++) {
      resolveGame(`${region}-R32-${i}`);
      const g = gm.get(`${region}-R32-${i}`)!;
      setSlot(`${region}-S16-${Math.floor(i / 2)}`, i % 2 === 0 ? 'teamA' : 'teamB', g.winner);
    }
    for (let i = 0; i < 2; i++) {
      resolveGame(`${region}-S16-${i}`);
      const g = gm.get(`${region}-S16-${i}`)!;
      setSlot(`${region}-E8-0`, i === 0 ? 'teamA' : 'teamB', g.winner);
    }
    resolveGame(`${region}-E8-0`);
  });

  // E8 → F4: East/West → F4-0; South/Midwest → F4-1
  setSlot('F4-0', 'teamA', gm.get('East-E8-0')!.winner);
  setSlot('F4-0', 'teamB', gm.get('West-E8-0')!.winner);
  setSlot('F4-1', 'teamA', gm.get('South-E8-0')!.winner);
  setSlot('F4-1', 'teamB', gm.get('Midwest-E8-0')!.winner);
  resolveGame('F4-0');
  resolveGame('F4-1');

  // F4 → Championship
  setSlot('Championship-0', 'teamA', gm.get('F4-0')!.winner);
  setSlot('Championship-0', 'teamB', gm.get('F4-1')!.winner);
  resolveGame('Championship-0');

  return [...gm.values()];
}

// ─── Narrative generator ──────────────────────────────────────────────────────
function generateNarrative(a: MockTeam, b: MockTeam, pWin: number): InterrogatorData {
  const tempoDiff = Math.abs(a.tempo - b.tempo);
  let styleClash = '';
  if (tempoDiff > 6) {
    const slow = a.tempo < b.tempo ? a : b;
    const fast = a.tempo < b.tempo ? b : a;
    styleClash = `⚡ TEMPO CLASH: ${fast.name} pushes pace (${fast.tempo.toFixed(0)}) vs ${slow.name} (${slow.tempo.toFixed(0)}). Fewer possessions = higher variance.`;
  }

  const upsetTeam = a.seed > b.seed ? a : b;
  const favTeam = a.seed > b.seed ? b : a;
  const upsetProb = a.seed > b.seed ? pWin : 1 - pWin;
  let narrative = '', recommendation = '';

  if (upsetProb > 0.40) {
    narrative = `🚨 UPSET ALERT: (${upsetTeam.seed}) ${upsetTeam.name} has a ${(upsetProb * 100).toFixed(0)}% shot. ${upsetTeam.playStyle}. ${favTeam.name}'s luck metric ${favTeam.luck.toFixed(2)} suggests regression.`;
    recommendation = `HIGH-VALUE LEVERAGE: Pick ${upsetTeam.name} in large pools for differentiation.`;
  } else if (Math.abs(pWin - 0.5) < 0.08) {
    narrative = `⚖️ COIN FLIP: ${a.name} (${a.playStyle}) vs ${b.name} (${b.playStyle}). Comparable efficiency profiles.`;
    recommendation = `POOL STRATEGY: Take the lower seed for leverage in large pools.`;
  } else {
    const fav = pWin > 0.5 ? a : b;
    const dog = pWin > 0.5 ? b : a;
    narrative = `✅ CLEAR FAVORITE: (${fav.seed}) ${fav.name} over (${dog.seed}) ${dog.name}. ${fav.playStyle}. Coach ${fav.coachName} (${fav.coachTourneyRecord} in March).`;
    recommendation = `CHALK PICK: ${fav.name} is the safe choice.`;
  }

  const radarA = [
    Math.min(1, (a.adj_oe - 95) / 30),
    Math.min(1, (105 - a.adj_de) / 20),
    Math.min(1, a.tempo / 80),
    Math.min(1, parseCoachWins(a.coachTourneyRecord) / 30),
    Math.min(1, a.sos / 10),
  ];
  const radarB = [
    Math.min(1, (b.adj_oe - 95) / 30),
    Math.min(1, (105 - b.adj_de) / 20),
    Math.min(1, b.tempo / 80),
    Math.min(1, parseCoachWins(b.coachTourneyRecord) / 30),
    Math.min(1, b.sos / 10),
  ];

  return { teamA: a, teamB: b, pWin, narrative, recommendation, styleClash, radarA, radarB };
}

// ─── Radar Chart ─────────────────────────────────────────────────────────────
function RadarChart({ dataA, dataB }: { dataA: number[]; dataB: number[] }) {
  const categories = ['OFFENSE', 'DEFENSE', 'TEMPO', 'EXPERIENCE', 'SOS'];
  const cx = 100, cy = 100, r = 70;
  const getPoint = (i: number, val: number) => {
    const angle = (Math.PI * 2 * i) / 5 - Math.PI / 2;
    return { x: cx + Math.cos(angle) * r * val, y: cy + Math.sin(angle) * r * val };
  };
  const pathA = dataA.map((v, i) => getPoint(i, v)).map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + 'Z';
  const pathB = dataB.map((v, i) => getPoint(i, v)).map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + 'Z';

  return (
    <svg viewBox="0 0 200 200" style={{ width: '100%', maxWidth: '200px' }}>
      {[0.25, 0.5, 0.75, 1].map(scale => (
        <polygon key={scale} points={Array.from({ length: 5 }, (_, i) => getPoint(i, scale)).map(p => `${p.x},${p.y}`).join(' ')}
          fill="none" stroke="rgba(255,107,53,0.1)" strokeWidth="0.5" />
      ))}
      {categories.map((_, i) => {
        const p = getPoint(i, 1);
        return <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="rgba(255,107,53,0.15)" strokeWidth="0.5" />;
      })}
      <path d={pathA} fill="rgba(255,107,53,0.15)" stroke="#ff6b35" strokeWidth="1.5" />
      <path d={pathB} fill="rgba(52,152,219,0.15)" stroke="#3498db" strokeWidth="1.5" />
      {categories.map((cat, i) => {
        const p = getPoint(i, 1.25);
        return <text key={cat} x={p.x} y={p.y} fill="rgba(245,240,232,0.5)" fontSize="5" textAnchor="middle" dominantBaseline="middle">{cat}</text>;
      })}
    </svg>
  );
}

// ─── Mini Game Cell ───────────────────────────────────────────────────────────
interface GameCellProps {
  game: BracketGame;
  height: number;
  isSelected: boolean;
  onPickWinner: (gameId: string, winner: MockTeam) => void;
  onSelect: (gameId: string) => void;
}

function MiniGameCell({ game, height, isSelected, onPickWinner, onSelect }: GameCellProps) {
  const { teamA, teamB, winner, winProb, isUserPick } = game;
  const spread = teamA && teamB ? estimateSpread(teamA, teamB) : 0;

  return (
    <div
      style={{
        height,
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${isSelected ? 'rgba(255,107,53,0.4)' : 'rgba(255,107,53,0.07)'}`,
        borderRadius: '3px',
        overflow: 'hidden',
        background: isSelected ? 'rgba(255,107,53,0.04)' : 'rgba(0,0,0,0.2)',
        cursor: 'pointer',
        position: 'relative',
        marginBottom: '2px',
        flexShrink: 0,
      }}
      onClick={() => onSelect(game.id)}
    >
      {/* TeamA row (top half) */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          padding: '0 5px',
          gap: '3px',
          background: winner === teamA ? 'rgba(46,204,113,0.07)' : 'transparent',
          borderBottom: '1px solid rgba(255,107,53,0.05)',
          cursor: teamA ? 'pointer' : 'default',
          minHeight: 0,
        }}
        onClick={(e) => { if (!teamA) return; e.stopPropagation(); onPickWinner(game.id, teamA); }}
      >
        <span style={{ fontSize: '7px', color: teamA?.seed && teamA.seed <= 4 ? '#ff6b35' : 'rgba(255,255,255,0.3)', minWidth: '10px', textAlign: 'center', fontWeight: 600 }}>
          {teamA?.seed ?? '—'}
        </span>
        <span style={{
          fontSize: '8px',
          color: winner === teamA ? '#2ecc71' : teamA ? 'rgba(245,240,232,0.75)' : 'rgba(255,255,255,0.2)',
          fontWeight: winner === teamA ? 700 : 400,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
        }}>
          {teamA?.name ?? 'TBD'}
        </span>
        {teamA && teamB && (
          <span style={{ fontSize: '6px', color: 'rgba(255,255,255,0.25)', flexShrink: 0 }}>
            {(winProb * 100).toFixed(0)}%
          </span>
        )}
      </div>

      {/* TeamB row (bottom half) */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          padding: '0 5px',
          gap: '3px',
          background: winner === teamB ? 'rgba(46,204,113,0.07)' : 'transparent',
          cursor: teamB ? 'pointer' : 'default',
          minHeight: 0,
        }}
        onClick={(e) => { if (!teamB) return; e.stopPropagation(); onPickWinner(game.id, teamB); }}
      >
        <span style={{ fontSize: '7px', color: teamB?.seed && teamB.seed <= 4 ? '#ff6b35' : 'rgba(255,255,255,0.3)', minWidth: '10px', textAlign: 'center', fontWeight: 600 }}>
          {teamB?.seed ?? '—'}
        </span>
        <span style={{
          fontSize: '8px',
          color: winner === teamB ? '#2ecc71' : teamB ? 'rgba(245,240,232,0.75)' : 'rgba(255,255,255,0.2)',
          fontWeight: winner === teamB ? 700 : 400,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
        }}>
          {teamB?.name ?? 'TBD'}
        </span>
        {teamA && teamB && (
          <span style={{ fontSize: '6px', color: 'rgba(255,255,255,0.25)', flexShrink: 0 }}>
            {((1 - winProb) * 100).toFixed(0)}%
          </span>
        )}
      </div>

      {/* Spread label (visible in taller cells) */}
      {height >= 72 && teamA && teamB && (
        <div style={{ position: 'absolute', bottom: 3, right: 5, fontSize: '6px', color: 'rgba(212,168,67,0.6)' }}>
          {spread >= 0 ? `${teamA.name.split(' ').pop()} -${Math.abs(spread)}` : `${teamB.name.split(' ').pop()} -${Math.abs(spread)}`}
        </div>
      )}

      {/* User pick badge */}
      {isUserPick && (
        <div style={{
          position: 'absolute', top: 2, right: 2,
          fontSize: '5px', color: '#d4a843',
          background: 'rgba(212,168,67,0.15)', padding: '1px 3px', borderRadius: '2px',
          letterSpacing: '0.04em', fontWeight: 700,
        }}>
          USER
        </div>
      )}
    </div>
  );
}

// ─── Region Panel ─────────────────────────────────────────────────────────────
interface RegionPanelProps {
  region: string;
  games: BracketGame[];
  selectedGameId: string | null;
  onPickWinner: (gameId: string, winner: MockTeam) => void;
  onSelectGame: (gameId: string) => void;
}

function RegionPanel({ region, games, selectedGameId, onPickWinner, onSelectGame }: RegionPanelProps) {
  const CELL_H = 34; // base height for R64 game cell

  const byRound = (round: RoundName) =>
    games.filter(g => g.round === round && g.region === region).sort((a, b) => a.pos - b.pos);

  const r64 = byRound('R64');
  const r32 = byRound('R32');
  const s16 = byRound('S16');
  const e8 = byRound('E8');

  const ROUND_COLS: { label: string; games: BracketGame[]; multiplier: number }[] = [
    { label: 'R64', games: r64, multiplier: 1 },
    { label: 'R32', games: r32, multiplier: 2 },
    { label: 'S16', games: s16, multiplier: 4 },
    { label: 'E8', games: e8, multiplier: 8 },
  ];

  return (
    <div style={{ background: 'rgba(26,18,8,0.6)', borderRadius: '8px', padding: '8px', border: '1px solid rgba(255,107,53,0.08)' }}>
      <div style={{ fontSize: '9px', color: '#ff6b35', fontWeight: 700, letterSpacing: '0.08em', marginBottom: '6px' }}>
        {region.toUpperCase()} REGION
      </div>
      <div style={{ display: 'flex', gap: '3px', alignItems: 'flex-start' }}>
        {ROUND_COLS.map(({ label, games: roundGames, multiplier }) => (
          <div key={label} style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: '6px', color: 'rgba(255,107,53,0.5)', textAlign: 'center', marginBottom: '3px', letterSpacing: '0.05em' }}>
              {label}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              {roundGames.map(game => (
                <MiniGameCell
                  key={game.id}
                  game={game}
                  height={CELL_H * multiplier}
                  isSelected={selectedGameId === game.id}
                  onPickWinner={onPickWinner}
                  onSelect={onSelectGame}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Large Game Cell (F4 / Championship) ─────────────────────────────────────
interface LargeGameCellProps {
  game: BracketGame;
  label: string;
  isSelected: boolean;
  onPickWinner: (gameId: string, winner: MockTeam) => void;
  onSelect: (gameId: string) => void;
}

function LargeGameCell({ game, label, isSelected, onPickWinner, onSelect }: LargeGameCellProps) {
  const { teamA, teamB, winner, winProb, isUserPick } = game;
  const accentColor = game.round === 'Championship' ? '#d4a843' : '#7b2fff';

  return (
    <div
      style={{
        background: 'rgba(26,18,8,0.8)',
        border: `1px solid ${isSelected ? accentColor : `${accentColor}33`}`,
        borderRadius: '8px',
        overflow: 'hidden',
        cursor: 'pointer',
        minWidth: '140px',
      }}
      onClick={() => onSelect(game.id)}
    >
      <div style={{ fontSize: '7px', color: accentColor, letterSpacing: '0.08em', textAlign: 'center', padding: '4px 8px 2px', fontWeight: 700 }}>
        {label.toUpperCase()}
      </div>
      {/* TeamA */}
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 8px',
          background: winner === teamA ? 'rgba(46,204,113,0.08)' : 'transparent',
          borderTop: `1px solid ${accentColor}22`,
          cursor: teamA ? 'pointer' : 'default',
        }}
        onClick={(e) => { if (!teamA) return; e.stopPropagation(); onPickWinner(game.id, teamA); }}
      >
        {teamA && <span style={{ fontSize: '8px', color: '#ff6b35', minWidth: '14px', textAlign: 'center', fontWeight: 700 }}>#{teamA.seed}</span>}
        <span style={{
          fontSize: '11px', fontWeight: winner === teamA ? 700 : 400,
          color: winner === teamA ? '#2ecc71' : teamA ? 'rgba(245,240,232,0.9)' : 'rgba(255,255,255,0.25)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
        }}>
          {teamA?.name ?? 'TBD'}
        </span>
        {teamA && teamB && (
          <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.4)' }}>{(winProb * 100).toFixed(0)}%</span>
        )}
      </div>
      {/* Divider with spread */}
      <div style={{ height: '1px', background: `${accentColor}22`, position: 'relative' }}>
        {teamA && teamB && (
          <span style={{
            position: 'absolute', left: '50%', top: '-8px', transform: 'translateX(-50%)',
            fontSize: '7px', color: `${accentColor}99`, background: 'rgba(26,18,8,0.9)',
            padding: '1px 4px', borderRadius: '2px',
          }}>
            vs
          </span>
        )}
      </div>
      {/* TeamB */}
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 8px',
          background: winner === teamB ? 'rgba(46,204,113,0.08)' : 'transparent',
          cursor: teamB ? 'pointer' : 'default',
        }}
        onClick={(e) => { if (!teamB) return; e.stopPropagation(); onPickWinner(game.id, teamB); }}
      >
        {teamB && <span style={{ fontSize: '8px', color: '#ff6b35', minWidth: '14px', textAlign: 'center', fontWeight: 700 }}>#{teamB.seed}</span>}
        <span style={{
          fontSize: '11px', fontWeight: winner === teamB ? 700 : 400,
          color: winner === teamB ? '#2ecc71' : teamB ? 'rgba(245,240,232,0.9)' : 'rgba(255,255,255,0.25)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
        }}>
          {teamB?.name ?? 'TBD'}
        </span>
        {teamA && teamB && (
          <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.4)' }}>{((1 - winProb) * 100).toFixed(0)}%</span>
        )}
      </div>
      {isUserPick && (
        <div style={{
          fontSize: '6px', color: '#d4a843', textAlign: 'center',
          padding: '2px', background: 'rgba(212,168,67,0.08)', letterSpacing: '0.05em',
        }}>
          USER PICK
        </div>
      )}
    </div>
  );
}

// ─── Matchup Interrogator Sidebar ─────────────────────────────────────────────
function Interrogator({
  data,
  wpaMap,
  setWpaMap,
  onClose,
}: {
  data: InterrogatorData;
  wpaMap: Record<string, number>;
  setWpaMap: React.Dispatch<React.SetStateAction<Record<string, number>>>;
  onClose: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 40, width: 0 }}
      animate={{ opacity: 1, x: 0, width: 300 }}
      exit={{ opacity: 0, x: 40, width: 0 }}
      style={{
        width: 300, flexShrink: 0,
        background: 'rgba(26, 18, 8, 0.97)',
        borderLeft: '1px solid rgba(255,107,53,0.15)',
        backdropFilter: 'blur(20px)',
        borderRadius: '12px',
        overflow: 'auto',
        maxHeight: 'calc(100vh - 80px)',
      }}
    >
      <div className="p-4">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <span style={{ fontSize: '10px', color: '#d4a843', letterSpacing: '0.08em', fontWeight: 700 }}>🔍 MATCHUP INTERROGATOR</span>
          <button onClick={onClose} style={{ color: 'var(--text-muted)', cursor: 'pointer', background: 'none', border: 'none', fontSize: '16px' }}>✕</button>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>#{data.teamA.seed} SEED</div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#ff6b35' }}>{data.teamA.name}</div>
            <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>{data.teamA.record}</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '18px', fontWeight: 800, color: '#d4a843' }}>{(data.pWin * 100).toFixed(0)}%</div>
            <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>WIN PROB</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>#{data.teamB.seed} SEED</div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#3498db' }}>{data.teamB.name}</div>
            <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>{data.teamB.record}</div>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '8px' }}>
          <RadarChart dataA={data.radarA} dataB={data.radarB} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', marginBottom: '12px' }}>
          {[{ name: data.teamA.name, color: '#ff6b35' }, { name: data.teamB.name, color: '#3498db' }].map(({ name, color }) => (
            <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
              <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>{name}</span>
            </div>
          ))}
        </div>

        <div className="glass-wood p-3 rounded-lg mb-3" style={{ borderLeft: '3px solid #d4a843' }}>
          <div style={{ fontSize: '9px', color: '#d4a843', letterSpacing: '0.06em', fontWeight: 600, marginBottom: '4px' }}>AI ANALYSIS</div>
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>{data.narrative}</div>
        </div>

        {data.styleClash && (
          <div className="glass-wood p-3 rounded-lg mb-3" style={{ borderLeft: '3px solid #e74c3c' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>{data.styleClash}</div>
          </div>
        )}

        <div className="glass-wood p-3 rounded-lg mb-3" style={{ borderLeft: '3px solid #2ecc71' }}>
          <div style={{ fontSize: '9px', color: '#2ecc71', letterSpacing: '0.06em', fontWeight: 600, marginBottom: '4px' }}>MODEL RECOMMENDATION</div>
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>{data.recommendation}</div>
        </div>

        <div className="glass-highlight p-3 rounded-lg mb-3" style={{ borderTop: '2px solid #3498db' }}>
          <div style={{ fontSize: '9px', color: '#3498db', letterSpacing: '0.06em', fontWeight: 600, marginBottom: '8px' }}>WIN PROBABILITY ADDED (WPA)</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {[data.teamA, data.teamB].map(team => (
              <div key={team.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <label htmlFor={`wpa-${team.name}`} style={{ fontSize: '10px', color: 'var(--text-muted)', width: '60px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{team.name}</label>
                <input
                  id={`wpa-${team.name}`}
                  type="range" min="-15" max="15"
                  value={wpaMap[team.name] ?? 0}
                  onChange={(e) => setWpaMap(prev => ({ ...prev, [team.name]: parseInt(e.target.value) }))}
                  style={{ flex: 1, margin: '0 8px' }}
                />
                <span style={{ fontSize: '10px', color: '#3498db', width: '24px', textAlign: 'right' }}>
                  {(wpaMap[team.name] ?? 0) > 0 ? '+' : ''}{wpaMap[team.name] ?? 0}
                </span>
              </div>
            ))}
          </div>
          <div style={{ fontSize: '8px', color: 'var(--text-secondary)', marginTop: '8px' }}>Simulate injuries or momentum factors.</div>
        </div>

        <div className="glass-wood p-3 rounded-lg">
          <div style={{ fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.06em', marginBottom: '6px' }}>HEAD-TO-HEAD</div>
          {[
            { label: 'Adj OE', a: data.teamA.adj_oe.toFixed(1), b: data.teamB.adj_oe.toFixed(1), aWins: data.teamA.adj_oe > data.teamB.adj_oe, neutral: false },
            { label: 'Adj DE', a: data.teamA.adj_de.toFixed(1), b: data.teamB.adj_de.toFixed(1), aWins: data.teamA.adj_de < data.teamB.adj_de, neutral: false },
            { label: 'Tempo', a: data.teamA.tempo.toFixed(0), b: data.teamB.tempo.toFixed(0), aWins: false, neutral: true },
            { label: 'SOS', a: data.teamA.sos.toFixed(1), b: data.teamB.sos.toFixed(1), aWins: data.teamA.sos > data.teamB.sos, neutral: false },
            { label: 'Coach W', a: data.teamA.coachTourneyRecord, b: data.teamB.coachTourneyRecord, aWins: parseCoachWins(data.teamA.coachTourneyRecord) > parseCoachWins(data.teamB.coachTourneyRecord), neutral: false },
          ].map(row => (
            <div key={row.label} style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '4px', padding: '3px 0', borderBottom: '1px solid rgba(255,107,53,0.05)' }}>
              <span style={{ fontSize: '10px', textAlign: 'right', fontWeight: !row.neutral && row.aWins ? 700 : 400, color: !row.neutral && row.aWins ? '#2ecc71' : 'var(--text-secondary)' }}>{row.a}</span>
              <span style={{ fontSize: '8px', color: 'var(--text-muted)', textAlign: 'center', minWidth: '40px' }}>{row.label}</span>
              <span style={{ fontSize: '10px', fontWeight: !row.neutral && !row.aWins ? 700 : 400, color: !row.neutral && !row.aWins ? '#2ecc71' : 'var(--text-secondary)' }}>{row.b}</span>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

// ─── Main FullBracket component ───────────────────────────────────────────────
export function FullBracket() {
  const [chaos, setChaos] = useState(0.5);
  const [wpaMap, setWpaMap] = useState<Record<string, number>>({});
  const [userPicks, setUserPicks] = useState<Map<string, MockTeam>>(new Map());
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [simData, setSimData] = useState<SimulateResponse | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);

  // Pre-sort teams by region using seed order
  const bracketByRegion = useMemo(() => {
    const result: Record<string, MockTeam[]> = {};
    REGIONS.forEach(region => {
      const regionTeams = MOCK_TEAMS.filter(t => t.region === region);
      const ordered: MockTeam[] = [];
      SEED_ORDER.forEach(seed => {
        const team = regionTeams.find(t => t.seed === seed);
        if (team) ordered.push(team);
      });
      result[region] = ordered;
    });
    return result;
  }, []);

  // Derived bracket — recomputed whenever chaos, wpaMap, or userPicks change
  const games = useMemo(
    () => computeBracket(bracketByRegion, userPicks, chaos, wpaMap),
    [bracketByRegion, userPicks, chaos, wpaMap],
  );

  const champion = useMemo(() => games.find(g => g.id === 'Championship-0')?.winner ?? null, [games]);

  // Derived interrogator for the selected game
  const interrogator = useMemo(() => {
    if (!selectedGameId) return null;
    const game = games.find(g => g.id === selectedGameId);
    if (!game?.teamA || !game?.teamB) return null;
    const pWin = computeWinProb(game.teamA, game.teamB, chaos, wpaMap);
    return generateNarrative(game.teamA, game.teamB, pWin);
  }, [selectedGameId, games, chaos, wpaMap]);

  const handlePickWinner = useCallback((gameId: string, winner: MockTeam) => {
    setUserPicks(prev => {
      const next = new Map(prev);
      // Toggle: clicking same winner again resets the pick
      const current = prev.get(gameId);
      if (current?.name === winner.name) {
        next.delete(gameId);
      } else {
        next.set(gameId, winner);
      }
      return next;
    });
  }, []);

  const handleSelectGame = useCallback((gameId: string) => {
    setSelectedGameId(prev => prev === gameId ? null : gameId);
  }, []);

  const handleSimulate = useCallback(async () => {
    setSimLoading(true);
    setSimError(null);
    try {
      const teamNames = MOCK_TEAMS.map(t => t.name);
      const result = await simulateBracket(teamNames, 1000);
      setSimData(result);
    } catch (err) {
      setSimError('Simulation failed — using cached data if available.');
      console.error('[BracketSimulator] simulateBracket error:', err);
    } finally {
      setSimLoading(false);
    }
  }, []);

  const handleResetPicks = useCallback(() => {
    setUserPicks(new Map());
    setWpaMap({});
    setSelectedGameId(null);
  }, []);

  const f4_0 = games.find(g => g.id === 'F4-0')!;
  const f4_1 = games.find(g => g.id === 'F4-1')!;
  const champGame = games.find(g => g.id === 'Championship-0')!;

  return (
    <div style={{ display: 'flex', gap: '12px', padding: '16px', maxWidth: '100%' }}>
      {/* ── Main bracket area ── */}
      <div style={{ flex: 1, minWidth: 0, overflow: 'auto' }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px', flexWrap: 'wrap' }}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ff6b35" strokeWidth="1.5" strokeLinecap="round">
            <path d="M 4 4 L 10 4 L 10 10" /><path d="M 4 20 L 10 20 L 10 14" />
            <path d="M 10 7 L 16 7 L 16 12" /><path d="M 10 17 L 16 17 L 16 12" />
            <path d="M 16 12 L 20 12" /><circle cx="20" cy="12" r="1.5" fill="#ff6b35" />
          </svg>
          <h1 style={{ fontFamily: 'var(--font-space-grotesk)', color: '#ff6b35', fontWeight: 700, fontSize: '18px', letterSpacing: '0.05em' }}>
            BRACKET ENGINE
          </h1>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px' }}>
            {(userPicks.size > 0 || Object.keys(wpaMap).length > 0) && (
              <button
                onClick={handleResetPicks}
                style={{ fontSize: '9px', color: '#e74c3c', background: 'rgba(231,76,60,0.1)', border: '1px solid rgba(231,76,60,0.3)', borderRadius: '4px', padding: '4px 8px', cursor: 'pointer' }}
              >
                RESET {userPicks.size > 0 ? `PICKS (${userPicks.size})` : 'WPA'}
              </button>
            )}
            <button
              onClick={handleSimulate}
              disabled={simLoading}
              style={{ fontSize: '9px', color: '#00f5ff', background: 'rgba(0,245,255,0.1)', border: '1px solid rgba(0,245,255,0.3)', borderRadius: '4px', padding: '4px 10px', cursor: simLoading ? 'wait' : 'pointer' }}
            >
              {simLoading ? 'SIMULATING…' : '⚡ RUN 1000 SIMS'}
            </button>
          </div>
        </div>
        {simError && (
          <div style={{ fontSize: '9px', color: '#e74c3c', marginBottom: '8px' }}>{simError}</div>
        )}

        {/* Chaos slider */}
        <div className="glass-highlight p-3 rounded-lg mb-4" style={{ borderTop: '2px solid #d4a843' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <span style={{ fontSize: '9px', color: '#d4a843', letterSpacing: '0.1em', fontWeight: 700 }}>🎛️ RISK & VARIANCE DIAL</span>
            <span style={{ fontSize: '11px', fontWeight: 700, color: chaos < 0.3 ? '#2ecc71' : chaos > 0.7 ? '#e74c3c' : '#d4a843' }}>
              {chaos < 0.3 ? '🏆 CHALK' : chaos > 0.7 ? '🔥 CHAOS' : '⚖️ VALUE'}
            </span>
          </div>
          <input
            type="range" min="0" max="100" value={chaos * 100}
            onChange={(e) => setChaos(parseInt(e.target.value) / 100)}
            style={{ width: '100%', height: '6px', borderRadius: '3px', appearance: 'none', background: `linear-gradient(90deg, #2ecc71 0%, #d4a843 50%, #e74c3c 100%)`, cursor: 'pointer' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '3px' }}>
            <span style={{ fontSize: '7px', color: '#2ecc71' }}>FAVORITES</span>
            <span style={{ fontSize: '7px', color: '#d4a843' }}>VALUE ZONE</span>
            <span style={{ fontSize: '7px', color: '#e74c3c' }}>CINDERELLAS</span>
          </div>
        </div>

        {/* Champion banner */}
        <AnimatePresence mode="wait">
          {champion && (
            <motion.div
              key={champion.name}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="glass-wood p-3 rounded-lg mb-4 text-center"
              style={{ border: '1px solid rgba(212,168,67,0.3)' }}
            >
              <div style={{ fontSize: '8px', color: '#d4a843', letterSpacing: '0.1em', fontWeight: 600 }}>🏆 PREDICTED CHAMPION</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#d4a843', margin: '2px 0' }}>{champion.name}</div>
              <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
                #{champion.seed} Seed • {getConferenceName(champion.conference)} • {champion.record}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Final Four + Championship ── */}
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '8px', color: '#7b2fff', letterSpacing: '0.1em', fontWeight: 700, marginBottom: '6px' }}>
            FINAL FOUR & CHAMPIONSHIP
          </div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start', flexWrap: 'wrap' }}>
            {f4_0 && (
              <LargeGameCell
                game={f4_0}
                label="Final Four — East/West"
                isSelected={selectedGameId === f4_0.id}
                onPickWinner={handlePickWinner}
                onSelect={handleSelectGame}
              />
            )}
            {champGame && (
              <LargeGameCell
                game={champGame}
                label="National Championship"
                isSelected={selectedGameId === champGame.id}
                onPickWinner={handlePickWinner}
                onSelect={handleSelectGame}
              />
            )}
            {f4_1 && (
              <LargeGameCell
                game={f4_1}
                label="Final Four — South/Midwest"
                isSelected={selectedGameId === f4_1.id}
                onPickWinner={handlePickWinner}
                onSelect={handleSelectGame}
              />
            )}
          </div>
        </div>

        {/* ── Regional Brackets (2×2 grid) ── */}
        <div style={{ fontSize: '8px', color: 'rgba(255,107,53,0.5)', letterSpacing: '0.08em', marginBottom: '8px' }}>
          CLICK A TEAM NAME TO PICK WINNER · CLICK GAME ROW TO OPEN INTERROGATOR
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
          {REGIONS.map(region => (
            <RegionPanel
              key={region}
              region={region}
              games={games}
              selectedGameId={selectedGameId}
              onPickWinner={handlePickWinner}
              onSelectGame={handleSelectGame}
            />
          ))}
        </div>

        {/* ── Monte Carlo Heatmap ── */}
        {simData && (
          <div className="glass-wood p-4 rounded-lg mt-4">
            <BracketHeatmap data={simData} />
          </div>
        )}
      </div>

      {/* ── Matchup Interrogator sidebar ── */}
      <AnimatePresence mode="wait">
        {interrogator && (
          <Interrogator
            data={interrogator}
            wpaMap={wpaMap}
            setWpaMap={setWpaMap}
            onClose={() => setSelectedGameId(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
