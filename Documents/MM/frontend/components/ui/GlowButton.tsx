'use client';
import { motion } from 'framer-motion';
import { ButtonHTMLAttributes } from 'react';
import { cn } from '@/lib/utils';

interface GlowButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'cyan' | 'violet' | 'green';
  loading?: boolean;
}

export function GlowButton({
  variant = 'cyan',
  loading = false,
  className,
  children,
  disabled,
  ...props
}: GlowButtonProps) {
  const colors = {
    cyan: 'border-[#00f5ff] text-[#00f5ff] shadow-[0_0_20px_rgba(0,245,255,0.4)]',
    violet: 'border-[#7b2fff] text-[#7b2fff] shadow-[0_0_20px_rgba(123,47,255,0.4)]',
    green: 'border-[#00ff88] text-[#00ff88] shadow-[0_0_20px_rgba(0,255,136,0.4)]',
  }[variant];

  return (
    <motion.button
      whileHover={disabled || loading ? {} : { scale: 1.04 }}
      whileTap={disabled || loading ? {} : { scale: 0.97 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      disabled={disabled || loading}
      className={cn(
        'relative px-6 py-2.5 rounded-lg border bg-transparent font-medium',
        'transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed',
        colors,
        className,
      )}
      {...(props as any)}
    >
      {loading ? (
        <span className="inline-flex items-center gap-2">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8v8z"
            />
          </svg>
          Loading…
        </span>
      ) : (
        children
      )}
    </motion.button>
  );
}
