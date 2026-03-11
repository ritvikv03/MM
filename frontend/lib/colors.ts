export const COLORS = {
  void_black: '#020408',
  obsidian: '#0a0f1a',
  cyan_core: '#00f5ff',
  violet_deep: '#7b2fff',
  tritium_green: '#00ff88',
  blood_red: '#ff2d55',
  amber_warn: '#ffb800',
} as const;

export type ColorKey = keyof typeof COLORS;

export const BLOOM = {
  cyan: { color: COLORS.cyan_core, intensity: 1.8, threshold: 0.6 },
  violet: { color: COLORS.violet_deep, intensity: 1.5, threshold: 0.6 },
  green: { color: COLORS.tritium_green, intensity: 1.6, threshold: 0.6 },
} as const;

export function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) throw new Error(`Invalid hex color: ${hex}`);
  return {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16),
  };
}
