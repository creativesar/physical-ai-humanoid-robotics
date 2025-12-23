/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./docs/**/*.{md,mdx}",
    "./blog/**/*.{md,mdx}",
    "./pages/**/*.{js,jsx,ts,tsx}",
    "./node_modules/@docusaurus/core/lib/**/*.js",
  ],
  theme: {
    extend: {
      colors: {
        // Dark theme colors
        'dark-bg': '#000000',
        'dark-bg-alt': '#0a0a0a',
        'dark-card': '#0f0f0f',
        // Neon/cyan accents
        'cyan': {
          400: '#00d0ff',
          500: '#00ffff',
          600: '#00d0ff',
          700: '#0088ff',
          800: '#0066cc',
        },
        'blue': {
          400: '#0088ff',
          500: '#0066cc',
          600: '#0044aa',
          700: '#002288',
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'noise': `url("data:image/svg+xml,%3Csvg viewBox='0 0 250 250' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
      },
      boxShadow: {
        'neon': '0 0 15px rgba(0, 208, 255, 0.5)',
        'neon-lg': '0 0 25px rgba(0, 208, 255, 0.7)',
        'neon-xl': '0 0 40px rgba(0, 208, 255, 0.9)',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(0, 208, 255, 0.4)' },
          '50%': { boxShadow: '0 0 0 10px rgba(0, 208, 255, 0)' },
        }
      }
    },
  },
  plugins: [],
}