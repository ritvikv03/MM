'use client';
import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import * as THREE from 'three';
import type { GameEdge, TeamNode } from '@/lib/api-types';

interface GameEdgeProps {
  edge: GameEdge;
  teamPositions: Map<string, TeamNode>;
}

export function GameEdgeLine({ edge, teamPositions }: GameEdgeProps) {
  const source = teamPositions.get(edge.source);
  const target = teamPositions.get(edge.target);

  const points = useMemo(() => {
    if (!source || !target) return null;
    return [
      new THREE.Vector3(source.x, source.y, source.z),
      new THREE.Vector3(target.x, target.y, target.z),
    ];
  }, [source, target]);

  // Must be declared before any early return to satisfy Rules of Hooks
  const color = useMemo(
    () => edge.home_win === true ? '#2ecc71' : edge.home_win === false ? '#e74c3c' : 'rgba(255,107,53,0.15)',
    [edge.home_win],
  );

  if (!points) return null;

  return (
    <Line
      points={points}
      color={color}
      lineWidth={0.5}
      transparent
      opacity={0.25}
    />
  );
}
