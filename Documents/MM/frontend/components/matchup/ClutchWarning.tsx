'use client';
import { motion } from 'framer-motion';

interface ClutchWarningProps {
  teamName: string;
}

export function ClutchWarning({ teamName }: ClutchWarningProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
      style={{
        background: 'rgba(255,183,0,0.1)',
        border: '1px solid rgba(255,183,0,0.4)',
        color: '#ffb800',
      }}
    >
      <span style={{ fontSize: '14px' }}>⚠</span>
      <span>
        <strong>{teamName}</strong> luck metric compressed — regression prior applied
      </span>
    </motion.div>
  );
}
