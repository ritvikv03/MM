import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './lib/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        void_black: '#020408',
        obsidian: '#0a0f1a',
        cyan_core: '#00f5ff',
        violet_deep: '#7b2fff',
        tritium_green: '#00ff88',
        blood_red: '#ff2d55',
        amber_warn: '#ffb800',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'breathe': 'breathe 2.5s ease-in-out infinite',
        'spin-slow': 'spin 8s linear infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1', filter: 'brightness(1)' },
          '50%': { opacity: '0.75', filter: 'brightness(1.4)' },
        },
        'breathe': {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.03)' },
        },
      },
    },
  },
  plugins: [],
};

export default config;
