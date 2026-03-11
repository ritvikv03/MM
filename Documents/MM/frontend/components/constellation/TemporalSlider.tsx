'use client';
import { motion } from 'framer-motion';

interface TemporalSliderProps {
  min: number;
  max: number;
  value: number;
  onChange: (value: number) => void;
  label?: string;
}

export function TemporalSlider({ min, max, value, onChange, label }: TemporalSliderProps) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;

  return (
    <div className="flex flex-col gap-1 w-full">
      {label && (
        <div className="flex justify-between text-xs">
          <span style={{ color: '#00f5ff' }}>{label}</span>
          <span style={{ color: '#888' }}>Day {value}</span>
        </div>
      )}
      <div className="relative h-2 rounded-full" style={{ background: 'rgba(0,245,255,0.1)' }}>
        <motion.div
          className="absolute top-0 left-0 h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: 'linear-gradient(90deg, #7b2fff, #00f5ff)',
          }}
          layoutId="slider-fill"
        />
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          style={{ zIndex: 1 }}
        />
      </div>
    </div>
  );
}
