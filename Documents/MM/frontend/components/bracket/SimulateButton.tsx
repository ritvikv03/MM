'use client';
import { motion } from 'framer-motion';

interface SimulateButtonProps {
  state: 'idle' | 'simulating' | 'settling' | 'heatmap';
  onSimulate: () => void;
  onReset: () => void;
  progress?: number; // 0–1
}

export function SimulateButton({
  state,
  onSimulate,
  onReset,
  progress = 0,
}: SimulateButtonProps) {
  const isRunning = state === 'simulating' || state === 'settling';
  const isDone = state === 'heatmap';

  const circumference = 2 * Math.PI * 20; // r=20
  const strokeDashoffset = circumference * (1 - progress);

  return (
    <div className="relative inline-flex items-center justify-center">
      {/* Progress ring */}
      {isRunning && (
        <svg
          className="absolute"
          width="64"
          height="64"
          style={{ transform: 'rotate(-90deg)' }}
        >
          <circle
            cx="32" cy="32" r="20"
            fill="none"
            stroke="rgba(0,245,255,0.15)"
            strokeWidth="2.5"
          />
          <motion.circle
            cx="32" cy="32" r="20"
            fill="none"
            stroke="#00f5ff"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 0.3 }}
          />
        </svg>
      )}

      <motion.button
        whileHover={isRunning ? {} : { scale: 1.05 }}
        whileTap={isRunning ? {} : { scale: 0.95 }}
        disabled={isRunning}
        onClick={isDone ? onReset : onSimulate}
        className="px-6 py-2.5 rounded-lg font-medium text-sm transition-colors"
        style={{
          background: isDone
            ? 'rgba(0,255,136,0.15)'
            : 'rgba(0,245,255,0.1)',
          border: `1px solid ${isDone ? '#00ff88' : '#00f5ff'}`,
          color: isDone ? '#00ff88' : '#00f5ff',
          opacity: isRunning ? 0.7 : 1,
          cursor: isRunning ? 'not-allowed' : 'pointer',
        }}
      >
        {state === 'idle' && 'Run Simulation'}
        {state === 'simulating' && 'Simulating…'}
        {state === 'settling' && 'Rendering…'}
        {state === 'heatmap' && 'Reset'}
      </motion.button>
    </div>
  );
}
