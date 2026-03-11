'use client';
import { useMemo } from 'react';
import { silvermanBandwidth, linspace, computeKDE } from '@/lib/d3-kde';

interface RidgelinePlotProps {
  homeSamples: number[];
  awaySamples: number[];
  homeTeam: string;
  awayTeam: string;
  width?: number;
  height?: number;
}

const W = 480;
const H = 180;
const MARGIN = { top: 20, right: 20, bottom: 30, left: 20 };
const INNER_W = W - MARGIN.left - MARGIN.right;
const INNER_H = H - MARGIN.top - MARGIN.bottom;

export function RidgelinePlot({
  homeSamples,
  awaySamples,
  homeTeam,
  awayTeam,
  width = W,
  height = H,
}: RidgelinePlotProps) {
  const { homePoints, awayPoints, gridX } = useMemo(() => {
    const grid = linspace(0, 1, 200);
    const homeBW = silvermanBandwidth(homeSamples);
    const awayBW = silvermanBandwidth(awaySamples);
    const homeD = computeKDE(homeSamples, grid, homeBW);
    const awayD = computeKDE(awaySamples, grid, awayBW);
    const maxD = Math.max(...homeD, ...awayD, 1e-6);

    // Scale to SVG coords
    const toX = (p: number) => MARGIN.left + p * INNER_W;
    const toY = (d: number) => MARGIN.top + INNER_H - (d / maxD) * INNER_H;

    const homePoints = grid.map((p, i) => `${toX(p)},${toY(homeD[i])}`).join(' ');
    const awayPoints = grid.map((p, i) => `${toX(p)},${toY(awayD[i])}`).join(' ');
    const gridX = grid;

    return { homePoints, awayPoints, gridX };
  }, [homeSamples, awaySamples]);

  // Build polygon paths (fill down to baseline)
  const baseY = MARGIN.top + INNER_H;
  const homePolygon = `${MARGIN.left},${baseY} ${homePoints} ${MARGIN.left + INNER_W},${baseY}`;
  const awayPolygon = `${MARGIN.left},${baseY} ${awayPoints} ${MARGIN.left + INNER_W},${baseY}`;

  // x-axis ticks at 0.1 intervals
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${W} ${H}`}
      style={{ overflow: 'visible' }}
    >
      <defs>
        <linearGradient id="homeGrad" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor="#00f5ff" stopOpacity="0.0" />
          <stop offset="50%" stopColor="#00f5ff" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#00f5ff" stopOpacity="0.0" />
        </linearGradient>
        <linearGradient id="awayGrad" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor="#7b2fff" stopOpacity="0.0" />
          <stop offset="50%" stopColor="#7b2fff" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#7b2fff" stopOpacity="0.0" />
        </linearGradient>
      </defs>

      {/* Away fill */}
      <polygon points={awayPolygon} fill="url(#awayGrad)" />
      {/* Home fill */}
      <polygon points={homePolygon} fill="url(#homeGrad)" />

      {/* Away line */}
      <polyline points={awayPoints} fill="none" stroke="#7b2fff" strokeWidth="1.5" />
      {/* Home line */}
      <polyline points={homePoints} fill="none" stroke="#00f5ff" strokeWidth="1.5" />

      {/* 50% reference line */}
      <line
        x1={MARGIN.left + INNER_W / 2}
        y1={MARGIN.top}
        x2={MARGIN.left + INNER_W / 2}
        y2={MARGIN.top + INNER_H}
        stroke="#ffb800"
        strokeWidth="0.75"
        strokeDasharray="4 4"
        opacity={0.5}
      />

      {/* X-axis baseline */}
      <line
        x1={MARGIN.left}
        y1={MARGIN.top + INNER_H}
        x2={MARGIN.left + INNER_W}
        y2={MARGIN.top + INNER_H}
        stroke="rgba(255,255,255,0.15)"
        strokeWidth="1"
      />

      {/* Axis ticks */}
      {ticks.map((t) => (
        <g key={t} transform={`translate(${MARGIN.left + t * INNER_W}, ${MARGIN.top + INNER_H})`}>
          <line y2="4" stroke="rgba(255,255,255,0.3)" />
          <text
            y="14"
            textAnchor="middle"
            fill="rgba(255,255,255,0.4)"
            fontSize="9"
          >
            {(t * 100).toFixed(0)}%
          </text>
        </g>
      ))}

      {/* Legend */}
      <circle cx={MARGIN.left + 8} cy={MARGIN.top + 8} r="4" fill="#00f5ff" />
      <text x={MARGIN.left + 16} y={MARGIN.top + 12} fill="#00f5ff" fontSize="10">{homeTeam}</text>
      <circle cx={MARGIN.left + 8 + 100} cy={MARGIN.top + 8} r="4" fill="#7b2fff" />
      <text x={MARGIN.left + 16 + 100} y={MARGIN.top + 12} fill="#7b2fff" fontSize="10">{awayTeam}</text>
    </svg>
  );
}
