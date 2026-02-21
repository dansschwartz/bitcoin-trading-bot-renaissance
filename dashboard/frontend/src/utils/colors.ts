/** Color palettes for charts and regime visualization. */

export const REGIME_COLORS: Record<string, string> = {
  bullish: '#00d395',
  bull: '#00d395',
  bull_trending: '#00d395',
  bull_mean_reverting: '#06b6d4',
  neutral: '#3b82f6',
  neutral_sideways: '#3b82f6',
  bearish: '#ff4757',
  bear: '#ff4757',
  bear_trending: '#ff4757',
  bear_mean_reverting: '#f97316',
  high_volatility: '#fbbf24',
  low_volatility: '#a855f7',
  trending: '#00d395',
  mean_reverting: '#06b6d4',
  crisis: '#ff4757',
  unknown: '#6b7280',
};

export const ACTION_COLORS: Record<string, string> = {
  BUY: '#00d395',
  SELL: '#ff4757',
  HOLD: '#6b7280',
};

export const CHART_COLORS = [
  '#3b82f6', '#00d395', '#fbbf24', '#a855f7', '#ff4757',
  '#06b6d4', '#f97316', '#ec4899', '#84cc16', '#14b8a6',
];

export function pnlBgColor(value: number): string {
  if (value > 0) return 'rgba(0, 211, 149, 0.1)';
  if (value < 0) return 'rgba(255, 71, 87, 0.1)';
  return 'rgba(107, 114, 128, 0.1)';
}

export function regimeColor(regime: string | undefined | null): string {
  if (!regime) return '#6b7280';
  const lower = regime.toLowerCase();
  // Direct match first
  if (REGIME_COLORS[lower]) return REGIME_COLORS[lower];
  // Fuzzy match: check if any key is contained in the regime name
  for (const [key, color] of Object.entries(REGIME_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return '#6b7280';
}
