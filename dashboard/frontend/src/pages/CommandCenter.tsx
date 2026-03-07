import { useEffect, useState, useCallback } from 'react';
import PageShell from '../components/layout/PageShell';
import MetricCard from '../components/cards/MetricCard';
import EquityCurve from '../components/charts/EquityCurve';
import PriceChart from '../components/charts/PriceChart';
import AssetSummaryPanel from '../components/panels/AssetSummaryPanel';
import ActivityFeed from '../components/panels/ActivityFeed';
import SystemHealthBar from '../components/panels/SystemHealthBar';
import { api } from '../api';

// ─── Types for strategy data ─────────────────────────────────────────────────

interface StrategyData {
  dailyPnl: number;
  allTimePnl: number;
  winRate: number;
  tradeCount: number;
  tradeLabel: string;
  loaded: boolean;
}

const EMPTY: StrategyData = { dailyPnl: 0, allTimePnl: 0, winRate: 0, tradeCount: 0, tradeLabel: '', loaded: false };

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtDollar(n: number): string {
  const abs = Math.abs(n);
  const sign = n >= 0 ? '+' : '-';
  if (abs >= 1000) return `${sign}$${abs.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
  return `${sign}$${abs.toFixed(2)}`;
}

function fmtCount(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function pnlColor(n: number): string {
  if (n > 0) return 'text-accent-green';
  if (n < 0) return 'text-accent-red';
  return 'text-gray-400';
}

// ─── StrategyCard (inline) ───────────────────────────────────────────────────

function StrategyCard({ name, data }: { name: string; data: StrategyData }) {
  if (!data.loaded) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <p className="text-xs text-gray-500 uppercase tracking-wider">{name}</p>
        <p className="text-sm text-gray-600 mt-2">Loading...</p>
      </div>
    );
  }

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">{name}</p>
      <div className="space-y-1">
        <div className="flex justify-between items-baseline">
          <span className="text-xs text-gray-500">Today</span>
          <span className={`text-lg font-semibold font-mono ${pnlColor(data.dailyPnl)}`}>
            {fmtDollar(data.dailyPnl)}
          </span>
        </div>
        <div className="flex justify-between items-baseline">
          <span className="text-xs text-gray-500">All-time</span>
          <span className={`text-sm font-mono ${pnlColor(data.allTimePnl)}`}>
            {fmtDollar(data.allTimePnl)}
          </span>
        </div>
        <div className="border-t border-surface-3 pt-1 mt-1 flex justify-between text-xs text-gray-500">
          <span>{data.winRate.toFixed(1)}% WR</span>
          <span>{fmtCount(data.tradeCount)} {data.tradeLabel}</span>
        </div>
      </div>
    </div>
  );
}

// ─── CommandCenter ───────────────────────────────────────────────────────────

export default function CommandCenter() {
  const [arb, setArb] = useState<StrategyData>(EMPTY);
  const [pm, setPm] = useState<StrategyData>(EMPTY);
  const [ml, setMl] = useState<StrategyData>(EMPTY);
  const [straddle, setStraddle] = useState<StrategyData>(EMPTY);

  const fetchAll = useCallback(async () => {
    // Arbitrage
    api.arbSummary().then(d => {
      setArb({
        dailyPnl: d.daily_pnl_usd ?? 0,
        allTimePnl: d.total_profit_usd ?? 0,
        winRate: d.win_rate ?? 0,
        tradeCount: d.filled_trades ?? 0,
        tradeLabel: 'fills',
        loaded: true,
      });
    }).catch(() => {});

    // Polymarket
    api.pmOverview().then((d: any) => {
      setPm({
        dailyPnl: d.today_pnl ?? 0,
        allTimePnl: d.total_pnl ?? 0,
        winRate: d.win_rate ?? 0,
        tradeCount: d.total_bets ?? 0,
        tradeLabel: 'bets',
        loaded: true,
      });
    }).catch(() => {});

    // ML Trading — need two calls
    Promise.all([
      api.pnl('1D'),
      api.riskMetrics(),
    ]).then(([pnlData, risk]) => {
      setMl({
        dailyPnl: pnlData.total_pnl ?? 0,
        allTimePnl: risk.cumulative_pnl ?? 0,
        winRate: pnlData.win_rate ?? 0,
        tradeCount: pnlData.total_round_trips ?? 0,
        tradeLabel: 'trips',
        loaded: true,
      });
    }).catch(() => {});

    // Straddles — compute daily from hourly
    Promise.all([
      api.straddleStats(),
      api.straddleHourly(),
    ]).then(([stats, hourly]: [any, any[]]) => {
      const today = new Date().toISOString().slice(0, 10);
      const straddleDailyPnl = (hourly ?? [])
        .filter((h: any) => h.hour?.startsWith(today))
        .reduce((sum: number, h: any) => sum + (h.pnl_usd ?? 0), 0);
      setStraddle({
        dailyPnl: straddleDailyPnl,
        allTimePnl: stats.total_pnl_usd ?? 0,
        winRate: stats.win_rate ?? 0,
        tradeCount: stats.total ?? 0,
        tradeLabel: 'straddles',
        loaded: true,
      });
    }).catch(() => {});
  }, []);

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 10_000);
    return () => clearInterval(id);
  }, [fetchAll]);

  // Combined totals
  const allLoaded = arb.loaded && pm.loaded && ml.loaded && straddle.loaded;
  const combinedDaily = arb.dailyPnl + pm.dailyPnl + ml.dailyPnl + straddle.dailyPnl;
  const combinedAllTime = arb.allTimePnl + pm.allTimePnl + ml.allTimePnl + straddle.allTimePnl;

  return (
    <PageShell title="Command Center" subtitle="Real-time operational overview">
      {/* System Health Bar */}
      <SystemHealthBar />

      {/* Combined Totals Row */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          title="Today P&L (All Strategies)"
          value={allLoaded ? fmtDollar(combinedDaily) : '--'}
          subtitle={allLoaded ? '4 active strategies' : 'loading...'}
          valueColor={allLoaded ? pnlColor(combinedDaily) : 'text-gray-100'}
        />
        <MetricCard
          title="All-Time P&L"
          value={allLoaded ? fmtDollar(combinedAllTime) : '--'}
          subtitle={allLoaded ? 'cumulative across all strategies' : 'loading...'}
          valueColor={allLoaded ? pnlColor(combinedAllTime) : 'text-gray-100'}
        />
      </div>

      {/* Strategy Breakdown Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StrategyCard name="Arbitrage" data={arb} />
        <StrategyCard name="Polymarket" data={pm} />
        <StrategyCard name="ML Trading" data={ml} />
        <StrategyCard name="Straddles" data={straddle} />
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
