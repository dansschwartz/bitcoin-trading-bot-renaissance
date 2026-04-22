/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          0: '#0a0e17',
          1: '#111827',
          2: '#1a2235',
          3: '#243049',
        },
        accent: {
          green: '#00d395',
          red: '#ff4757',
          blue: '#3b82f6',
          yellow: '#fbbf24',
          purple: '#a855f7',
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
    },
  },
  plugins: [],
};
