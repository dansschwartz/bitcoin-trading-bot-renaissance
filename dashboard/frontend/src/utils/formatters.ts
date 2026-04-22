/** Formatting utilities for dashboard display values. */

export function formatCurrency(value: number, decimals = 2): string {
  const abs = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(2)}K`;
  return `${sign}$${abs.toFixed(decimals)}`;
}

export function formatPercent(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatBps(value: number): string {
  return `${(value * 10_000).toFixed(1)} bps`;
}

export function formatNumber(value: number, decimals = 4): string {
  return value.toFixed(decimals);
}

export function formatTimestamp(iso: string): string {
  if (!iso) return '--';
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '--';
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '--';
  }
}

export function formatDate(iso: string): string {
  if (!iso) return '--';
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '--';
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  } catch {
    return '--';
  }
}

export function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function pnlColor(value: number): string {
  if (value > 0) return 'text-accent-green';
  if (value < 0) return 'text-accent-red';
  return 'text-gray-400';
}

export function pnlSign(value: number): string {
  return value >= 0 ? '+' : '';
}
