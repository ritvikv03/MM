'use client';
import { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { TeamNodeMesh } from './TeamNode';
import { ConferenceNodeMesh } from './ConferenceNode';
import { GameEdgeLine } from './GameEdge';
import { ConferenceEdgeLine } from './ConferenceEdge';
import { BasketballCourt3D } from './BasketballCourt3D';
import type { GraphResponse, TeamNode } from '@/lib/api-types';

interface ConstellationCanvasProps {
  graph: GraphResponse;
  selectedTeam?: string | null;
  onSelectTeam?: (team: TeamNode) => void;
}

export function ConstellationCanvas({
  graph,
  selectedTeam,
  onSelectTeam,
}: ConstellationCanvasProps) {
  const teamMap = useMemo(
    () => new Map(graph.teams.map((t) => [t.id, t])),
    [graph.teams],
  );
  const conferenceMap = useMemo(
    () => new Map(graph.conferences.map((c) => [c.id, c])),
    [graph.conferences],
  );

  return (
    <div style={{ width: '100%', height: '100%', background: '#020408' }}>
      <Canvas
        camera={{ position: [0, 14, 10], fov: 55 }}
        gl={{ antialias: true, alpha: false }}
      >
        <Suspense fallback={null}>
          {/* Overhead basketball court lighting */}
          <ambientLight intensity={0.3} />
          <hemisphereLight color="#c8a040" groundColor="#020408" intensity={0.6} />
          <pointLight position={[0, 10, 0]} intensity={0.8} color="#fff8e0" />
          <pointLight position={[10, 8, 5]} intensity={0.3} color="#00f5ff" />
          <pointLight position={[-10, 8, -5]} intensity={0.2} color="#7b2fff" />

          {/* Basketball court floor */}
          <BasketballCourt3D />

          {/* Conference edges (member_of) */}
          {graph.conference_edges.map((edge) => (
            <ConferenceEdgeLine
              key={`ce-${edge.source}-${edge.target}`}
              edge={edge}
              teamPositions={teamMap}
              conferencePositions={conferenceMap}
            />
          ))}

          {/* Game edges — limit to 150 for performance */}
          {graph.games.slice(0, 150).map((edge, i) => (
            <GameEdgeLine key={`ge-${edge.source}-${edge.target}-${i}`} edge={edge} teamPositions={teamMap} />
          ))}

          {/* Conference nodes */}
          {graph.conferences.map((node) => (
            <ConferenceNodeMesh key={node.id} node={node} />
          ))}

          {/* Team nodes */}
          {graph.teams.map((node) => (
            <TeamNodeMesh
              key={node.id}
              node={node}
              onSelect={onSelectTeam}
              isSelected={selectedTeam === node.id}
            />
          ))}

          <OrbitControls
            enablePan={false}
            minDistance={6}
            maxDistance={28}
            autoRotate
            autoRotateSpeed={0.3}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}
