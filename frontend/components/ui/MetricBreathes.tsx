'use client';
import { motion } from 'framer-motion';

interface MetricBreathesProps {
  scale?: number;
  duration?: number;
  children: React.ReactNode;
}

export function MetricBreathes({
  scale = 1.03,
  duration = 2.5,
  children,
}: MetricBreathesProps) {
  return (
    <motion.div
      animate={{ scale: [1, scale, 1] }}
      transition={{ duration, repeat: Infinity, ease: 'easeInOut' }}
    >
      {children}
    </motion.div>
  );
}
