import { COLORS } from './colors';

export interface LayoutNode {
  id: string;
  x: number;
  y: number;
  z: number;
}

export function fibonacciSphere(
  n: number,
  radius: number = 5,
): Array<{ x: number; y: number; z: number }> {
  const points: Array<{ x: number; y: number; z: number }> = [];
  const goldenRatio = (1 + Math.sqrt(5)) / 2;
  for (let i = 0; i < n; i++) {
    const theta = Math.acos(1 - (2 * (i + 0.5)) / n);
    const phi = (2 * Math.PI * i) / goldenRatio;
    points.push({
      x: radius * Math.sin(theta) * Math.cos(phi),
      y: radius * Math.sin(theta) * Math.sin(phi),
      z: radius * Math.cos(theta),
    });
  }
  return points;
}

export function marginToColor(margin: number): number {
  // Returns 0xRRGGBB integer based on predicted margin
  if (margin > 10) return 0x00ff88; // tritium_green — strong favorite
  if (margin > 3) return 0x00f5ff; // cyan_core — moderate favorite
  if (margin > -3) return 0xffb800; // amber_warn — toss-up
  if (margin > -10) return 0x7b2fff; // violet_deep — moderate underdog
  return 0xff2d55; // blood_red — heavy underdog
}

const CONFERENCE_COLORS: Record<string, number> = {
  ACC: 0x00f5ff,
  'Big Ten': 0x7b2fff,
  'Big 12': 0x00ff88,
  SEC: 0xff2d55,
  'Pac-12': 0xffb800,
};

export function conferenceColor(conference: string): number {
  return CONFERENCE_COLORS[conference] ?? 0x888888;
}

export interface ComputeLayoutResult {
  conferencePositions: Array<{ id: string; x: number; y: number; z: number }>;
  teamPositions: Array<{ id: string; x: number; y: number; z: number }>;
}

export function computeLayout(
  conferenceIds: string[],
  teamsByConference: Record<string, string[]>,
  conferenceRadius: number = 5,
  teamSpread: number = 1.5,
): ComputeLayoutResult {
  const conferenceSpherePoints = fibonacciSphere(conferenceIds.length, conferenceRadius);
  const conferencePositions = conferenceIds.map((id, i) => ({
    id,
    ...conferenceSpherePoints[i],
  }));
  const confPosMap = new Map(conferencePositions.map((c) => [c.id, c]));

  const teamPositions: Array<{ id: string; x: number; y: number; z: number }> = [];
  for (const [confId, teamIds] of Object.entries(teamsByConference)) {
    const center = confPosMap.get(confId) ?? { x: 0, y: 0, z: 0 };
    const clusterPoints = fibonacciSphere(teamIds.length, teamSpread);
    teamIds.forEach((teamId, i) => {
      teamPositions.push({
        id: teamId,
        x: center.x + clusterPoints[i].x,
        y: center.y + clusterPoints[i].y,
        z: center.z + clusterPoints[i].z,
      });
    });
  }
  return { conferencePositions, teamPositions };
}

// Re-export COLORS reference to satisfy import (used by graph-layout consumers)
export { COLORS };
