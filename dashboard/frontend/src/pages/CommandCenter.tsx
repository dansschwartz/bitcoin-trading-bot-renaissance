import PageShell from '../components/layout/PageShell';
import PnLCard from '../components/cards/PnLCard';
import MetricCard from '../components/cards/MetricCard';
import RegimeCard from '../components/cards/RegimeCard';
import EquityCurve from '../components/charts/EquityCurve';
import PriceChart from '../components/charts/PriceChart';
import OpenPositionsTable from '../components/tables/OpenPositionsTable';
import ActivityFeed from '../components/panels/ActivityFeed';
import { useDashboard } from '../context/DashboardContext';
import { formatCurrency, formatNumber } from '../utils/formatters';

export default function CommandCenter() {
  const { state } = useDashboard();
  const { status, pnl } = state;

  return (
    <PageShell title="Command Center" subtitle="Real-time operational overview">
      {/* Metric Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <PnLCard />
        <RegimeCard />
        <MetricCard
          title="Cycles Today"
          value={status?.cycle_count ?? '--'}
          subtitle={`${status?.product_ids?.length ?? 0} assets tracked`}
        />
        <MetricCard
          title="Avg Slippage"
          value={pnl ? `${(pnl.avg_slippage * 100).toFixed(3)}%` : '--'}
          subtitle={`${pnl?.total_trades ?? 0} trades (24h)`}
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
        <OpenPositionsTable />
        <ActivityFeed />
      </div>
    </PageShell>
  );
}
