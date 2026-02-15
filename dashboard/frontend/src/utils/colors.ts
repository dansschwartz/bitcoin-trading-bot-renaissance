/** Color palettes for charts and regime visualization. */

export const REGIME_COLORS: Record<string, string> = {
  bullish: '#00d395',
  bull: '#00d395',
  neutral: '#3b82f6',
  bearish: '#ff4757',
  bear: '#ff4757',
  high_volatility: '#fbbf24',
  low_volatility: '#a855f7',
  trending: '#00d395',
  mean_reverting: '#3b82f6',
  crisis: '#ff4757',
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
  return REGIME_COLORS[lower] || '#6b7280';
}
