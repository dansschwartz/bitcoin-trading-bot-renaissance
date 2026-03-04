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

interface SpraySnapshot {
  total_sprayed: number;
  total_open: number;
  today_pnl_usd: number;
  today_tokens: number;
  budget?: { deployed_usd: number; total_capital: number; deployed_pct: number };
}

export default function CommandCenter() {
  const { state } = useDashboard();
  const { status, pnl } = state;
  const [spray, setSpray] = useState<SpraySnapshot | null>(null);

  useEffect(() => {
    api.sprayStatus().then(d => setSpray(d as unknown as SpraySnapshot)).catch(() => {});
    const id = setInterval(() => {
      api.sprayStatus().then(d => setSpray(d as unknown as SpraySnapshot)).catch(() => {});
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
          title="Tokens Sprayed"
          value={spray ? `${spray.today_tokens ?? 0}` : '--'}
          subtitle={`${spray?.total_open ?? 0} open | ${(spray?.total_sprayed ?? 0).toLocaleString()} lifetime`}
        />
        <MetricCard
          title="Deployed Capital"
          value={spray?.budget ? `$${spray.budget.deployed_usd.toFixed(0)}` : '--'}
          subtitle={spray?.budget ? `${spray.budget.deployed_pct.toFixed(0)}% of $${spray.budget.total_capital.toFixed(0)}` : `${pnl?.total_trades ?? 0} trades (24h)`}
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
