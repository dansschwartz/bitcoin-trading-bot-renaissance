import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import ExposurePanel from '../components/panels/ExposurePanel';
import RiskGatewayLog from '../components/panels/RiskGatewayLog';
import AlertsPanel from '../components/panels/AlertsPanel';
import DevilTrackerPanel from '../components/panels/DevilTrackerPanel';
import MetricCard from '../components/cards/MetricCard';
import ConditionalPanel from '../components/shared/ConditionalPanel';
import { api } from '../api';
import type { RiskMetrics } from '../types';
import { formatCurrency, formatPercent } from '../utils/formatters';

export default function Risk() {
  const [metrics, setMetrics] = useState<RiskMetrics | null>(null);

  useEffect(() => {
    api.riskMetrics().then(setMetrics).catch(() => {});
    const id = setInterval(() => api.riskMetrics().then(setMetrics).catch(() => {}), 15_000);
    return () => clearInterval(id);
  }, []);

  return (
    <PageShell title="Risk Management" subtitle="Exposure monitoring, risk gateway, and alerts">
      {/* Risk metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          title="Current Equity"
          value={metrics ? formatCurrency(metrics.current_equity) : '--'}
          subtitle={metrics ? `Initial: ${formatCurrency(metrics.initial_capital)}` : ''}
          valueColor={metrics ? (metrics.current_equity >= metrics.initial_capital ? 'text-accent-green' : 'text-accent-red') : 'text-gray-100'}
        />
        <MetricCard
          title="Max Drawdown"
          value={metrics ? formatPercent(metrics.max_drawdown) : '--'}
          valueColor={metrics && metrics.max_drawdown > 0.05 ? 'text-accent-red' : 'text-gray-100'}
        />
        <MetricCard
          title="Peak Equity"
          value={metrics ? formatCurrency(metrics.peak_equity) : '--'}
        />
        <MetricCard
          title="Sharpe Ratio"
          value={metrics?.sharpe_ratio != null ? metrics.sharpe_ratio.toFixed(2) : '--'}
          valueColor={metrics ? (metrics.sharpe_ratio >= 1 ? 'text-accent-green' : metrics.sharpe_ratio >= 0 ? 'text-gray-100' : 'text-accent-red') : 'text-gray-100'}
          subtitle={metrics ? `${metrics.total_trading_days} trading days` : ''}
        />
      </div>

      {/* Secondary metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          title="Cumulative P&L"
          value={metrics ? formatCurrency(metrics.cumulative_pnl) : '--'}
          valueColor={metrics ? (metrics.cumulative_pnl >= 0 ? 'text-accent-green' : 'text-accent-red') : 'text-gray-100'}
        />
        <MetricCard
          title="Unrealized P&L"
          value={metrics ? formatCurrency(metrics.unrealized_pnl) : '--'}
          valueColor={metrics ? (metrics.unrealized_pnl >= 0 ? 'text-accent-green' : 'text-accent-red') : 'text-gray-100'}
        />
        <MetricCard
          title="Max Consec. Losses"
          value={metrics?.max_consecutive_losses ?? '--'}
          valueColor={metrics && metrics.max_consecutive_losses >= 5 ? 'text-accent-red' : 'text-gray-100'}
        />
        <MetricCard
          title="Trading Days"
          value={metrics?.total_trading_days ?? '--'}
        />
      </div>

      {/* Exposure + Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <ExposurePanel />
        <AlertsPanel />
      </div>

      {/* Cost Leakage (Devil Tracker) */}
      <DevilTrackerPanel />

      {/* Gateway Log */}
      <ConditionalPanel flag="risk_gateway" fallback={<RiskGatewayLog />}>
        <RiskGatewayLog />
      </ConditionalPanel>
    </PageShell>
  );
}
