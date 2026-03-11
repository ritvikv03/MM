'use client';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface TeamDropZoneProps {
  side: 'home' | 'away';
  teamName?: string;
  adjOE?: number;
  adjDE?: number;
  onClick?: () => void;
  className?: string;
}

export function TeamDropZone({
  side,
  teamName,
  adjOE,
  adjDE,
  onClick,
  className,
}: TeamDropZoneProps) {
  const accentColor = side === 'home' ? '#00f5ff' : '#7b2fff';
  const label = side === 'home' ? 'HOME' : 'AWAY';

  return (
    <motion.div
      onClick={onClick}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={cn('cursor-pointer rounded-xl p-4 select-none', className)}
      style={{
        background: 'rgba(10,15,26,0.7)',
        border: `1px solid ${accentColor}44`,
        backdropFilter: 'blur(20px)',
        boxShadow: teamName ? `0 0 20px ${accentColor}22` : 'none',
      }}
    >
      <div className="text-xs mb-1" style={{ color: `${accentColor}99` }}>
        {label}
      </div>
      {teamName ? (
        <>
          <div className="text-lg font-bold" style={{ color: accentColor }}>
            {teamName}
          </div>
          {adjOE !== undefined && adjDE !== undefined && (
            <div className="flex gap-3 mt-2 text-xs" style={{ color: 'rgba(255,255,255,0.5)' }}>
              <span>ORtg <span style={{ color: '#00ff88' }}>{adjOE.toFixed(1)}</span></span>
              <span>DRtg <span style={{ color: '#ff2d55' }}>{adjDE.toFixed(1)}</span></span>
            </div>
          )}
        </>
      ) : (
        <div className="text-sm" style={{ color: 'rgba(255,255,255,0.3)' }}>
          Click to select team
        </div>
      )}
    </motion.div>
  );
}
