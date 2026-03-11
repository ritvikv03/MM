'use client';
import { AnimatePresence, motion } from 'framer-motion';

interface SlotNumberProps {
  value: number;
  decimals?: number;
  className?: string;
}

export function SlotNumber({ value, decimals = 1, className }: SlotNumberProps) {
  const formatted = value.toFixed(decimals);
  return (
    <span
      className={className}
      style={{ display: 'inline-flex', overflow: 'hidden', verticalAlign: 'bottom' }}
    >
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          key={formatted}
          initial={{ y: '100%', opacity: 0 }}
          animate={{ y: '0%', opacity: 1 }}
          exit={{ y: '-100%', opacity: 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          style={{ display: 'inline-block' }}
        >
          {formatted}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}
