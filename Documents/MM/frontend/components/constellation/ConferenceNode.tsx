'use client';
import { useRef, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Icosahedron, Html } from '@react-three/drei';
import * as THREE from 'three';
import type { ConferenceNode } from '@/lib/api-types';

interface ConferenceNodeProps {
  node: ConferenceNode;
}

export function ConferenceNodeMesh({ node }: ConferenceNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const color = useMemo(
    () => new THREE.Color(`#${node.color.toString(16).padStart(6, '0')}`),
    [node.color],
  );

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += delta * 0.3;
      meshRef.current.rotation.y += delta * 0.2;
    }
  });

  return (
    <group position={[node.x, node.y, node.z]}>
      <Icosahedron
        ref={meshRef}
        args={[0.35, 1]}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={hovered ? 0.6 : 0.2}
          wireframe={true}
        />
      </Icosahedron>
      {hovered && (
        <Html distanceFactor={10} center>
          <div style={{
            background: 'rgba(10,15,26,0.9)',
            border: '1px solid #7b2fff',
            borderRadius: '6px',
            padding: '4px 8px',
            color: '#7b2fff',
            fontSize: '11px',
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
          }}>
            {node.name}
          </div>
        </Html>
      )}
    </group>
  );
}
