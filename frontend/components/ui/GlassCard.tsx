'use client';
import { motion, HTMLMotionProps } from 'framer-motion';
import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

interface GlassCardProps extends HTMLMotionProps<'div'> {
  variant?: 'default' | 'cyan' | 'violet' | 'green';
  children: React.ReactNode;
}

export const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
  ({ variant = 'default', className, children, ...props }, ref) => {
    const glassClass = {
      default: 'glass',
      cyan: 'glass-cyan',
      violet: 'glass-violet',
      green: 'glass',
    }[variant];

    return (
      <motion.div
        ref={ref}
        className={cn(glassClass, 'p-4', className)}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        {...props}
      >
        {children}
      </motion.div>
    );
  },
);

GlassCard.displayName = 'GlassCard';
