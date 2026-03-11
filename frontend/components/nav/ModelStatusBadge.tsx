'use client';
import { motion } from 'framer-motion';

interface ModelStatusBadgeProps {
  status: 'online' | 'offline' | 'loading';
  version?: string;
}

const STATUS_CONFIG = {
  online:  { color: '#2ecc71', label: 'Model Online' },
  offline: { color: '#e74c3c', label: 'Model Offline' },
  loading: { color: '#d4a843', label: 'Loading…' },
};

export function ModelStatusBadge({ status, version = 'v1.0' }: ModelStatusBadgeProps) {
  const { color, label } = STATUS_CONFIG[status];
  return (
    <div className="flex items-center gap-2 px-3 py-1 rounded-full"
      style={{ background: `${color}18`, border: `1px solid ${color}44` }}
    >
      <motion.div
        animate={{ opacity: [1, 0.3, 1] }}
        transition={{ duration: 1.5, repeat: Infinity }}
        style={{ width: '6px', height: '6px', borderRadius: '50%', background: color }}
      />
      <span style={{ fontSize: '11px', color, fontFamily: 'monospace' }}>
        {label} {version}
      </span>
    </div>
  );
}
