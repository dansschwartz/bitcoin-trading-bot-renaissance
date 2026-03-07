import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import PnLCard from '../components/cards/PnLCard';
import MetricCard from '../components/cards/MetricCard';
import RegimeCard from '../components/cards/RegimeCard';
import EquityCurve from '../components/charts/EquityCurve';
import PriceChart from '../components/charts/PriceChart';
import AssetSummaryPanel from '../components/panels/AssetSummaryPanel';
import ActivityFeed from '../components/panels/ActivityFeed';
import SystemHealthBar from '../components/panels/SystemHealthBar';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../api';

interface ArbSnapshot {
  total_profit_usd: number;
  filled_trades: number;
  win_rate: number;
  daily_pnl_usd: number;
}

export default function CommandCenter() {
  const { state } = useDashboard();
  const { pnl } = state;
  const [arb, setArb] = useState<ArbSnapshot | null>(null);

  useEffect(() => {
    api.arbSummary().then(d => setArb(d as unknown as ArbSnapshot)).catch(() => {});
    const id = setInterval(() => {
      api.arbSummary().then(d => setArb(d as unknown as ArbSnapshot)).catch(() => {});
    }, 10_000);
    return () => clearInterval(id);
  }, []);

  return (
    <PageShell title="Command Center" subtitle="Real-time operational overview">
      {/* System Health Bar */}
      <SystemHealthBar />

      {/* Metric Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <PnLCard />
        <RegimeCard />
        <MetricCard
          title="Arb P&L"
          value={arb ? `$${(arb.total_profit_usd ?? 0).toFixed(2)}` : '--'}
          subtitle={arb ? `${arb.filled_trades ?? 0} fills | ${(arb.win_rate ?? 0).toFixed(0)}% win` : 'loading...'}
        />
        <MetricCard
          title="Arb Today"
          value={arb ? `$${(arb.daily_pnl_usd ?? 0).toFixed(2)}` : '--'}
          subtitle={`${pnl?.total_trades ?? 0} ML trades (24h)`}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="lg:col-span-2">
          <EquityCurve />
        </div>
        <PriceChart />
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <AssetSummaryPanel compact />
        <ActivityFeed />
      </div>
    </PageShell>
  );
}
