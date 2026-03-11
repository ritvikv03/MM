'use client';
import { useRef, useMemo, useEffect } from 'react';
import * as THREE from 'three';

/**
 * Draws an NCAA basketball court onto a canvas and returns a Three.js CanvasTexture.
 * Court dimensions: 94ft × 50ft → mapped to PlaneGeometry(22, 11.7) in Three.js units.
 */
function buildCourtTexture(): THREE.CanvasTexture {
  const W = 940, H = 500; // canvas pixels, maintains ~94:50 ratio
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d')!;

  // Hardwood floor
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, '#c8a040');
  grad.addColorStop(1, '#d4aa50');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // Court line style
  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.lineWidth = 3;

  // Boundary
  ctx.strokeRect(20, 20, W - 40, H - 40);

  // Half-court line
  ctx.beginPath();
  ctx.moveTo(W / 2, 20);
  ctx.lineTo(W / 2, H - 20);
  ctx.stroke();

  // Center circle (radius ≈ 6ft on 94ft court → 940px wide → ~60px radius)
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, 60, 0, Math.PI * 2);
  ctx.stroke();

  // Center dot
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, 8, 0, Math.PI * 2);
  ctx.fill();

  // Paint (key): 19ft×12ft on 94ft court. Scale: 940/94=10px/ft.
  // Paint width = 120px, height = 190px, centered horizontally
  const paintW = 120, paintH = 190;

  // Left end paint + FT circle
  ctx.strokeRect(20, H / 2 - paintW / 2, paintH, paintW);
  ctx.beginPath();
  ctx.arc(20 + paintH, H / 2, paintW / 2, -Math.PI / 2, Math.PI / 2);
  ctx.stroke();
  // Restricted arc (left)
  ctx.beginPath();
  ctx.arc(20 + 50, H / 2, 40, -Math.PI / 2, Math.PI / 2);
  ctx.stroke();

  // Right end paint + FT circle
  ctx.strokeRect(W - 20 - paintH, H / 2 - paintW / 2, paintH, paintW);
  ctx.beginPath();
  ctx.arc(W - 20 - paintH, H / 2, paintW / 2, Math.PI / 2, -Math.PI / 2);
  ctx.stroke();
  // Restricted arc (right)
  ctx.beginPath();
  ctx.arc(W - 20 - 50, H / 2, 40, Math.PI / 2, -Math.PI / 2);
  ctx.stroke();

  // 3-point arcs (radius ≈ 235px scaled) with corner straights
  const arcR = 210;
  const cornerY = 75; // distance from sideline for corner 3pt

  // Left 3pt
  ctx.beginPath();
  ctx.moveTo(20, H / 2 - cornerY);
  ctx.lineTo(20 + 90, H / 2 - cornerY);
  const leftArcAngle = Math.acos((H / 2 - cornerY - H / 2) / arcR);
  ctx.arc(20 + 50, H / 2, arcR, -leftArcAngle, leftArcAngle);
  ctx.lineTo(20, H / 2 + cornerY);
  ctx.stroke();

  // Right 3pt
  ctx.beginPath();
  ctx.moveTo(W - 20, H / 2 - cornerY);
  ctx.lineTo(W - 20 - 90, H / 2 - cornerY);
  ctx.arc(W - 20 - 50, H / 2, arcR, Math.PI - leftArcAngle, Math.PI + leftArcAngle);
  ctx.lineTo(W - 20, H / 2 + cornerY);
  ctx.stroke();

  // Backboards
  ctx.strokeStyle = 'rgba(255,140,0,0.9)';
  ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(60, H / 2 - 30); ctx.lineTo(60, H / 2 + 30); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(W - 60, H / 2 - 30); ctx.lineTo(W - 60, H / 2 + 30); ctx.stroke();

  return new THREE.CanvasTexture(canvas);
}

export function BasketballCourt3D() {
  const meshRef = useRef<THREE.Mesh>(null);

  const texture = useMemo(() => {
    if (typeof window === 'undefined') return null;
    return buildCourtTexture();
  }, []);

  useEffect(() => {
    return () => {
      texture?.dispose();
      if (meshRef.current) {
        (meshRef.current.geometry as THREE.BufferGeometry).dispose();
        (meshRef.current.material as THREE.Material).dispose();
      }
    };
  }, [texture]);

  if (!texture) return null;

  return (
    <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]}>
      <planeGeometry args={[22, 11.7]} />
      <meshStandardMaterial map={texture} roughness={0.6} metalness={0.1} />
    </mesh>
  );
}
