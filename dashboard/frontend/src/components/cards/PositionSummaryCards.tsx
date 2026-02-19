import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { PositionSummary } from '../../types';
import MetricCard from './MetricCard';
import { formatCurrency, pnlColor } from '../../utils/formatters';

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}

export default function PositionSummaryCards() {
  const [summary, setSummary] = useState<PositionSummary | null>(null);

  useEffect(() => {
    api.positionSummary().then(setSummary).catch(() => {});
    const id = setInterval(() => {
      api.positionSummary().then(setSummary).catch(() => {});
    }, 30_000);
    return () => clearInterval(id);
  }, []);

  if (!summary || summary.total_closed === 0) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
      <MetricCard
        title="Win Rate"
        value={`${(summary.win_rate * 100).toFixed(1)}%`}
        subtitle={`${summary.wins}W / ${summary.losses}L`}
        valueColor={summary.win_rate >= 0.5 ? 'text-accent-green' : 'text-accent-red'}
      />
      <MetricCard
        title="Total Realized P&L"
        value={formatCurrency(summary.total_realized_pnl)}
        subtitle={`${summary.total_closed} round trips`}
        valueColor={pnlColor(summary.total_realized_pnl)}
      />
      <MetricCard
        title="Avg Win"
        value={formatCurrency(summary.avg_win)}
        valueColor="text-accent-green"
      />
      <MetricCard
        title="Avg Loss"
        value={formatCurrency(summary.avg_loss)}
        valueColor="text-accent-red"
      />
      <MetricCard
        title="Best Trade"
        value={formatCurrency(summary.largest_win)}
        valueColor="text-accent-green"
      />
      <MetricCard
        title="Worst Trade"
        value={formatCurrency(summary.largest_loss)}
        subtitle={`Avg hold: ${formatDuration(summary.avg_hold_seconds)}`}
        valueColor="text-accent-red"
      />
    </div>
  );
}
