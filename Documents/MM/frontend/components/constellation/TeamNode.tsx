'use client';
import { useRef, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Html } from '@react-three/drei';
import * as THREE from 'three';
import type { TeamNode } from '@/lib/api-types';

interface TeamNodeProps {
  node: TeamNode;
  onSelect?: (node: TeamNode) => void;
  isSelected?: boolean;
}

export function TeamNodeMesh({ node, onSelect, isSelected = false }: TeamNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const color = useMemo(
    () => new THREE.Color(`#${node.color.toString(16).padStart(6, '0')}`),
    [node.color],
  );
  const emissiveIntensity = isSelected ? 1.2 : hovered ? 0.8 : 0.3;
  const scale = isSelected ? 1.4 : hovered ? 1.2 : 1.0;

  useFrame((_, delta) => {
    if (meshRef.current && (hovered || isSelected)) {
      meshRef.current.rotation.y += delta * 0.5;
    }
  });

  return (
    <group position={[node.x, node.y, node.z]}>
      <Sphere
        ref={meshRef}
        args={[0.15, 16, 16]}
        scale={scale}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        onClick={() => onSelect?.(node)}
      >
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity}
          metalness={0.1}
          roughness={0.3}
        />
      </Sphere>
      {(hovered || isSelected) && (
        <Html distanceFactor={10} center>
          <div style={{
            background: 'rgba(10,15,26,0.9)',
            border: `1px solid #00f5ff`,
            borderRadius: '6px',
            padding: '4px 8px',
            color: '#00f5ff',
            fontSize: '11px',
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
          }}>
            {node.name}
            {node.seed && <span style={{ color: '#ffb800', marginLeft: '4px' }}>#{node.seed}</span>}
          </div>
        </Html>
      )}
    </group>
  );
}
