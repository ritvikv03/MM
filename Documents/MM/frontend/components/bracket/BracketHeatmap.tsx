'use client';
import type { SimulateResponse } from '@/lib/api-types';
import { probToHeatColor, ROUNDS } from '@/lib/bracket-utils';

interface BracketHeatmapProps {
  data: SimulateResponse;
}

const ROUND_LABELS: Record<string, string> = {
  R64: 'R64',
  R32: 'R32',
  S16: 'S16',
  E8: 'E8',
  F4: 'F4',
  Championship: 'CHAMP',
};

export function BracketHeatmap({ data }: BracketHeatmapProps) {
  // Sort teams by Championship probability descending
  const sorted = [...data.advancements].sort(
    (a, b) => (b.advancement_probs['Championship'] ?? 0) - (a.advancement_probs['Championship'] ?? 0),
  );

  if (sorted.length === 0) {
    return <div style={{ color: 'rgba(245,240,232,0.4)', fontSize: '11px', padding: '12px' }}>No simulation data available.</div>;
  }

  const cellW = 80;
  const cellH = 24;
  const labelW = 140;
  const headerH = 28;
  const svgW = labelW + ROUNDS.length * cellW + 4;
  const svgH = headerH + sorted.length * cellH + 4;

  return (
    <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: '500px' }}>
      <div style={{ fontSize: '9px', color: '#d4a843', letterSpacing: '0.08em', fontWeight: 600, marginBottom: '6px' }}>
        MONTE CARLO ADVANCEMENT HEATMAP
      </div>
      <svg width={svgW} height={svgH} style={{ display: 'block' }}>
        {/* Column headers */}
        {ROUNDS.map((round, ci) => (
          <text
            key={round}
            x={labelW + ci * cellW + cellW / 2}
            y={headerH - 8}
            fill="rgba(255,107,53,0.7)"
            fontSize={8}
            textAnchor="middle"
            fontFamily="var(--font-space-grotesk)"
            letterSpacing="0.06em"
          >
            {ROUND_LABELS[round] ?? round}
          </text>
        ))}

        {/* Data rows */}
        {sorted.map((team, ri) => {
          const y = headerH + ri * cellH;
          return (
            <g key={team.team}>
              {/* Team label */}
              <text
                x={labelW - 6}
                y={y + cellH / 2 + 3}
                fill="rgba(245,240,232,0.8)"
                fontSize={9}
                textAnchor="end"
                fontFamily="monospace"
              >
                {team.team.length > 18 ? team.team.slice(0, 17) + '…' : team.team}
              </text>

              {/* Round cells */}
              {ROUNDS.map((round, ci) => {
                const prob = team.advancement_probs[round] ?? 0;
                const bg = probToHeatColor(prob);
                return (
                  <g key={round}>
                    <rect
                      x={labelW + ci * cellW + 1}
                      y={y + 1}
                      width={cellW - 2}
                      height={cellH - 2}
                      fill={bg}
                      fillOpacity={Math.max(0.2, prob * 0.9)}
                      rx={2}
                    />
                    <text
                      x={labelW + ci * cellW + cellW / 2}
                      y={y + cellH / 2 + 3}
                      fill={prob >= 0.15 ? '#fff' : 'rgba(245,240,232,0.35)'}
                      fontSize={8}
                      textAnchor="middle"
                      fontFamily="monospace"
                    >
                      {prob >= 0.01 ? `${(prob * 100).toFixed(0)}%` : '—'}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
