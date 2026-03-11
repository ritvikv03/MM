'use client';
import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import * as THREE from 'three';
import type { ConferenceEdge, TeamNode, ConferenceNode } from '@/lib/api-types';

interface ConferenceEdgeProps {
  edge: ConferenceEdge;
  teamPositions: Map<string, TeamNode>;
  conferencePositions: Map<string, ConferenceNode>;
}

export function ConferenceEdgeLine({
  edge,
  teamPositions,
  conferencePositions,
}: ConferenceEdgeProps) {
  const source = teamPositions.get(edge.source);
  const target = conferencePositions.get(edge.target);

  const points = useMemo(() => {
    if (!source || !target) return null;
    return [
      new THREE.Vector3(source.x, source.y, source.z),
      new THREE.Vector3(target.x, target.y, target.z),
    ];
  }, [source, target]);

  if (!points) return null;

  return (
    <Line
      points={points}
      color="#7b2fff"
      lineWidth={0.3}
      transparent
      opacity={0.12}
      dashed
      dashSize={0.1}
      gapSize={0.1}
    />
  );
}
