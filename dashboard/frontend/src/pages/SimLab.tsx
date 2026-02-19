import { useEffect, useState, useMemo, useCallback } from 'react';
import PageShell from '../components/layout/PageShell';
import MetricCard from '../components/cards/MetricCard';
import { api } from '../api';
import type { PnLSummary, RiskMetrics, BacktestRun, BacktestComparison } from '../types';
import { formatCurrency, formatPercent, pnlColor } from '../utils/formatters';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';

// ─── Helpers ──────────────────────────────────────────────────────────────

const CHART_TOOLTIP_STYLE = {
  backgroundColor: '#1a2235',
  border: '1px solid #243049',
  borderRadius: 8,
  fontSize: 12,
  color: '#e5e7eb',
};

type SortField = 'id' | 'timestamp' | 'total_trades' | 'realized_pnl' | 'sharpe_ratio' | 'max_drawdown' | 'win_rate';
type SortDir = 'asc' | 'desc';

/** Determine if a value diverges from a reference by more than 2 standard deviations. */
function isDivergent(live: number, backtest: number, stdFactor = 0.15): boolean {
  // Use 15% of the backtest value as a rough proxy for 1 sigma when we lack a
  // full distribution.  A difference > 2x that threshold is flagged.
  if (backtest === 0) return Math.abs(live) > 0.01;
  const sigma = Math.abs(backtest) * stdFactor;
  return Math.abs(live - backtest) > 2 * sigma;
}

function formatDateShort(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric', year: '2-digit' });
  } catch {
    return iso;
  }
}

function formatDateTime(iso: string): string {
  try {
    const d = new Date(iso);
    return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  } catch {
    return iso;
  }
}

// ─── Section 1: Performance Dashboard ─────────────────────────────────────

function PerformanceDashboard() {
  const [pnlAll, setPnlAll] = useState<PnLSummary | null>(null);
  const [pnlWeek, setPnlWeek] = useState<PnLSummary | null>(null);
  const [risk, setRisk] = useState<RiskMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.pnl('ALL').catch(() => null),
      api.pnl('1W').catch(() => null),
      api.riskMetrics().catch(() => null),
    ]).then(([all, week, r]) => {
      setPnlAll(all);
      setPnlWeek(week);
      setRisk(r);
      setLoading(false);
    });

    const id = setInterval(() => {
      api.pnl('ALL').then(setPnlAll).catch(() => {});
      api.pnl('1W').then(setPnlWeek).catch(() => {});
      api.riskMetrics().then(setRisk).catch(() => {});
    }, 30_000);
    return () => clearInterval(id);
  }, []);

  // Use proper Sharpe ratio from risk metrics (annualized)
  const sharpeAll = useMemo(() => {
    if (!risk) return null;
    return risk.sharpe_ratio ?? null;
  }, [risk]);

  // Bar chart data: current period vs prior
  const comparisonData = useMemo(() => {
    if (!pnlAll || !pnlWeek) return [];
    return [
      { metric: 'P&L', current: pnlWeek.realized_pnl, allTime: pnlAll.realized_pnl },
      { metric: 'Win Rate', current: pnlWeek.win_rate * 100, allTime: pnlAll.win_rate * 100 },
      { metric: 'Trades', current: pnlWeek.total_trades, allTime: pnlAll.total_trades },
    ];
  }, [pnlAll, pnlWeek]);

  if (loading) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-8">
        <div className="flex items-center justify-center gap-3 text-gray-500">
          <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span className="text-sm">Loading performance data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
        <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        Live Performance Dashboard
      </h2>

      {/* Metric cards: All-Time row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          title="Strategy Return"
          value={pnlAll ? formatCurrency(pnlAll.realized_pnl) : '--'}
          subtitle="All time"
          valueColor={pnlAll ? (pnlAll.realized_pnl >= 0 ? 'text-accent-green' : 'text-accent-red') : 'text-gray-100'}
        />
        <MetricCard
          title="Sharpe Ratio"
          value={sharpeAll != null ? sharpeAll.toFixed(2) : '--'}
          subtitle="Annualized (365d)"
          valueColor={sharpeAll != null && sharpeAll >= 1 ? 'text-accent-green' : 'text-gray-100'}
        />
        <MetricCard
          title="Max Drawdown"
          value={risk ? formatPercent(risk.max_drawdown) : '--'}
          subtitle="Peak-to-trough"
          valueColor={risk && risk.max_drawdown > 0.05 ? 'text-accent-red' : 'text-gray-100'}
        />
        <MetricCard
          title="Win Rate"
          value={pnlAll ? formatPercent(pnlAll.win_rate) : '--'}
          subtitle={pnlAll ? `${pnlAll.winning_round_trips ?? 0}/${pnlAll.total_round_trips ?? 0} round-trips` : ''}
          valueColor={pnlAll && pnlAll.win_rate >= 0.5 ? 'text-accent-green' : 'text-gray-100'}
        />
      </div>

      {/* Metric cards: This Week row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          title="Weekly Return"
          value={pnlWeek ? formatCurrency(pnlWeek.realized_pnl) : '--'}
          subtitle="This week"
          valueColor={pnlWeek ? (pnlWeek.realized_pnl >= 0 ? 'text-accent-green' : 'text-accent-red') : 'text-gray-100'}
        />
        <MetricCard
          title="Unrealized P&L"
          value={risk ? formatCurrency(risk.unrealized_pnl) : '--'}
          subtitle="Open positions"
          valueColor={risk ? (risk.unrealized_pnl >= 0 ? 'text-accent-green' : 'text-accent-red') : 'text-gray-100'}
        />
        <MetricCard
          title="Total Trades"
          value={pnlAll?.total_trades ?? '--'}
          subtitle="All time"
        />
        <MetricCard
          title="Avg Slippage"
          value={pnlAll ? formatPercent(pnlAll.avg_slippage) : '--'}
          subtitle="All time"
          valueColor={pnlAll && pnlAll.avg_slippage > 0.005 ? 'text-accent-red' : 'text-gray-100'}
        />
      </div>

      {/* Period comparison bar chart */}
      {comparisonData.length > 0 && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">This Week vs All Time</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} margin={{ top: 5, right: 20, bottom: 5, left: 20 }}>
                <XAxis dataKey="metric" tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Bar dataKey="current" name="This Week" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="allTime" name="All Time" fill="#6b7280" radius={[4, 4, 0, 0]} fillOpacity={0.5} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Section 2: Backtest Runs Table ───────────────────────────────────────

interface RunsTableProps {
  runs: BacktestRun[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

function BacktestRunsTable({ runs, selectedId, onSelect }: RunsTableProps) {
  const [sortField, setSortField] = useState<SortField>('timestamp');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const handleSort = useCallback((field: SortField) => {
    setSortDir(prev => (sortField === field ? (prev === 'asc' ? 'desc' : 'asc') : 'desc'));
    setSortField(field);
  }, [sortField]);

  const sortedRuns = useMemo(() => {
    const copy = [...runs];
    copy.sort((a, b) => {
      let aVal: number | string = a[sortField];
      let bVal: number | string = b[sortField];
      if (sortField === 'timestamp') {
        aVal = new Date(a.timestamp).getTime();
        bVal = new Date(b.timestamp).getTime();
      }
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDir === 'asc' ? aVal - bVal : bVal - aVal;
      }
      return sortDir === 'asc'
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal));
    });
    return copy;
  }, [runs, sortField, sortDir]);

  const sortIcon = (field: SortField) => {
    if (sortField !== field) return <span className="text-gray-700 ml-1">&#x2195;</span>;
    return <span className="text-blue-400 ml-1">{sortDir === 'asc' ? '\u2191' : '\u2193'}</span>;
  };

  if (runs.length === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <div className="text-center py-16">
          <svg className="w-16 h-16 mx-auto text-gray-700 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
          <h3 className="text-lg font-medium text-gray-400 mb-2">No Backtest Runs</h3>
          <p className="text-sm text-gray-600 max-w-md mx-auto">
            Run <code className="bg-surface-2 px-1.5 py-0.5 rounded text-gray-300">replay_backtest.py</code> to
            generate backtest results, then they will appear here for comparison against live performance.
          </p>
          <div className="mt-4 text-xs text-gray-700 font-mono bg-surface-2 inline-block px-3 py-2 rounded-lg">
            python replay_backtest.py --config config.yaml
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
        <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
        Backtest Runs
        <span className="text-xs text-gray-600 font-normal">({runs.length} run{runs.length !== 1 ? 's' : ''})</span>
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-gray-500 border-b border-surface-3">
              <th className="text-left py-2 px-2 cursor-pointer select-none" onClick={() => handleSort('id')}>
                ID{sortIcon('id')}
              </th>
              <th className="text-left py-2 px-2 cursor-pointer select-none" onClick={() => handleSort('timestamp')}>
                Date{sortIcon('timestamp')}
              </th>
              <th className="text-right py-2 px-2 cursor-pointer select-none" onClick={() => handleSort('total_trades')}>
                Trades{sortIcon('total_trades')}
              </th>
              <th className="text-right py-2 px-2 cursor-pointer select-none" onClick={() => handleSort('realized_pnl')}>
                P&L{sortIcon('realized_pnl')}
              </th>
              <th className="text-right py-2 px-2 cursor-pointer select-none" onClick={() => handleSort('sharpe_ratio')}>
                Sharpe{sortIcon('sharpe_ratio')}
              </th>
              <th className="text-right py-2 px-2 cursor-pointer select-none" onClick={() => handleSort('max_drawdown')}>
                Max DD{sortIcon('max_drawdown')}
              </th>
              <th className="text-right py-2 px-2 cursor-pointer select-none" onClick={() => handleSort('win_rate')}>
                Win Rate{sortIcon('win_rate')}
              </th>
              <th className="text-center py-2 px-2">Action</th>
            </tr>
          </thead>
          <tbody>
            {sortedRuns.map((run) => {
              const isSelected = run.id === selectedId;
              return (
                <tr
                  key={run.id}
                  className={`border-b border-surface-3/30 cursor-pointer transition-colors ${
                    isSelected ? 'bg-blue-500/10' : 'hover:bg-surface-2/50'
                  }`}
                  onClick={() => onSelect(run.id)}
                >
                  <td className="py-2 px-2 text-gray-300">#{run.id}</td>
                  <td className="py-2 px-2 text-gray-400">{formatDateTime(run.timestamp)}</td>
                  <td className="py-2 px-2 text-right text-gray-400">{run.total_trades}</td>
                  <td className={`py-2 px-2 text-right ${pnlColor(run.realized_pnl)}`}>
                    {formatCurrency(run.realized_pnl)}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">{run.sharpe_ratio.toFixed(2)}</td>
                  <td className="py-2 px-2 text-right text-accent-red">{formatPercent(run.max_drawdown)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">{formatPercent(run.win_rate)}</td>
                  <td className="py-2 px-2 text-center">
                    <button
                      className={`px-2 py-0.5 rounded text-xs transition-colors ${
                        isSelected
                          ? 'bg-blue-500 text-white'
                          : 'bg-surface-2 text-gray-400 hover:text-gray-200 hover:bg-surface-3'
                      }`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelect(run.id);
                      }}
                    >
                      {isSelected ? 'Selected' : 'Compare'}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Section 3: Live vs Backtest Comparison ──────────────────────────────

interface ComparisonProps {
  comparison: BacktestComparison;
  onClose: () => void;
}

interface ComparisonMetric {
  label: string;
  live: number;
  backtest: number;
  format: (v: number) => string;
  higherIsBetter: boolean;
}

function LiveVsBacktestComparison({ comparison, onClose }: ComparisonProps) {
  const { backtest, live } = comparison;

  const metrics: ComparisonMetric[] = useMemo(() => [
    {
      label: 'Return',
      live: live.pnl_summary.realized_pnl,
      backtest: backtest.realized_pnl,
      format: formatCurrency,
      higherIsBetter: true,
    },
    {
      label: 'Sharpe Ratio',
      live: live.risk_metrics.sharpe_ratio ?? 0,
      backtest: backtest.sharpe_ratio,
      format: (v: number) => v.toFixed(2),
      higherIsBetter: true,
    },
    {
      label: 'Max Drawdown',
      live: live.risk_metrics.max_drawdown,
      backtest: backtest.max_drawdown,
      format: formatPercent,
      higherIsBetter: false,
    },
    {
      label: 'Win Rate',
      live: live.pnl_summary.win_rate,
      backtest: backtest.win_rate,
      format: formatPercent,
      higherIsBetter: true,
    },
    {
      label: 'Total Trades',
      live: live.pnl_summary.total_trades,
      backtest: backtest.total_trades,
      format: (v: number) => v.toString(),
      higherIsBetter: true,
    },
  ], [backtest, live]);

  // Radar chart data for visual comparison
  const radarData = useMemo(() => {
    return metrics.map(m => {
      // Normalize values to 0-100 scale for the radar chart
      const maxVal = Math.max(Math.abs(m.live), Math.abs(m.backtest), 0.001);
      return {
        metric: m.label,
        Live: (Math.abs(m.live) / maxVal) * 100,
        Backtest: (Math.abs(m.backtest) / maxVal) * 100,
      };
    });
  }, [metrics]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <svg className="w-4 h-4 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Live vs Backtest #{backtest.id}
          <span className="text-xs text-gray-600 font-normal">
            ({formatDateShort(backtest.timestamp)})
          </span>
        </h2>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 transition-colors p-1 rounded hover:bg-surface-2"
          title="Close comparison"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Side-by-side metric cards */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="text-left py-2 px-3">Metric</th>
                <th className="text-right py-2 px-3">
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" />
                    Live
                  </span>
                </th>
                <th className="text-right py-2 px-3">
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-purple-400 inline-block" />
                    Backtest
                  </span>
                </th>
                <th className="text-right py-2 px-3">Delta</th>
                <th className="text-center py-2 px-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => {
                const delta = m.live - m.backtest;
                const divergent = isDivergent(m.live, m.backtest);
                const deltaPositive = m.higherIsBetter ? delta > 0 : delta < 0;
                const deltaColor = divergent
                  ? (deltaPositive ? 'text-accent-green' : 'text-accent-red')
                  : 'text-gray-500';

                return (
                  <tr
                    key={m.label}
                    className={`border-b border-surface-3/30 transition-colors ${
                      divergent ? 'bg-yellow-500/5' : ''
                    }`}
                  >
                    <td className="py-3 px-3 text-gray-300 font-medium">{m.label}</td>
                    <td className="py-3 px-3 text-right text-blue-300">{m.format(m.live)}</td>
                    <td className="py-3 px-3 text-right text-purple-300">{m.format(m.backtest)}</td>
                    <td className={`py-3 px-3 text-right ${deltaColor}`}>
                      {delta >= 0 ? '+' : ''}{m.format(delta)}
                    </td>
                    <td className="py-3 px-3 text-center">
                      {divergent ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-yellow-500/10 text-yellow-400 text-[10px] font-semibold">
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                          </svg>
                          {'>'}2sigma
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-500/10 text-green-400 text-[10px] font-semibold">
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                          </svg>
                          OK
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Radar chart + Divergence summary side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* Radar chart */}
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Performance Profile</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="#243049" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <PolarRadiusAxis tick={{ fontSize: 9, fill: '#6b7280' }} domain={[0, 100]} />
                <Radar name="Live" dataKey="Live" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} strokeWidth={2} />
                <Radar name="Backtest" dataKey="Backtest" stroke="#a855f7" fill="#a855f7" fillOpacity={0.1} strokeWidth={2} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Divergence summary */}
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Divergence Analysis</h3>
          <div className="space-y-3">
            {metrics.map((m) => {
              const divergent = isDivergent(m.live, m.backtest);
              const pctDelta = m.backtest !== 0
                ? ((m.live - m.backtest) / Math.abs(m.backtest)) * 100
                : 0;
              const barWidth = Math.min(Math.abs(pctDelta), 100);

              return (
                <div key={m.label} className={`rounded-lg p-3 ${divergent ? 'bg-yellow-500/5 border border-yellow-500/20' : 'bg-surface-2/50'}`}>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs text-gray-400">{m.label}</span>
                    <span className={`text-xs font-mono ${divergent ? 'text-yellow-400' : 'text-gray-500'}`}>
                      {pctDelta >= 0 ? '+' : ''}{pctDelta.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-surface-3 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        divergent
                          ? (pctDelta > 0 ? 'bg-yellow-400' : 'bg-red-400')
                          : 'bg-blue-400/40'
                      }`}
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                  {divergent && (
                    <p className="text-[10px] text-yellow-500/70 mt-1">
                      Live {m.label.toLowerCase()} deviates {'>'} 2 sigma from backtest
                    </p>
                  )}
                </div>
              );
            })}

            {/* Summary badge */}
            {(() => {
              const divergentCount = metrics.filter(m => isDivergent(m.live, m.backtest)).length;
              if (divergentCount === 0) {
                return (
                  <div className="mt-2 p-3 rounded-lg bg-green-500/5 border border-green-500/20 text-center">
                    <p className="text-xs text-green-400 font-medium">All metrics within expected range</p>
                    <p className="text-[10px] text-green-500/60 mt-0.5">Live performance aligns with backtest expectations</p>
                  </div>
                );
              }
              return (
                <div className="mt-2 p-3 rounded-lg bg-yellow-500/5 border border-yellow-500/20 text-center">
                  <p className="text-xs text-yellow-400 font-medium">
                    {divergentCount} metric{divergentCount > 1 ? 's' : ''} diverging from backtest
                  </p>
                  <p className="text-[10px] text-yellow-500/60 mt-0.5">
                    Review strategy parameters or market regime changes
                  </p>
                </div>
              );
            })()}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main SimLab Page ─────────────────────────────────────────────────────

export default function SimLab() {
  const [runs, setRuns] = useState<BacktestRun[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [comparison, setComparison] = useState<BacktestComparison | null>(null);
  const [compLoading, setCompLoading] = useState(false);
  const [compError, setCompError] = useState<string | null>(null);

  // Load backtest runs
  useEffect(() => {
    api.backtestRuns().then(setRuns).catch(() => {});
  }, []);

  // Load comparison when a run is selected
  useEffect(() => {
    if (selectedId == null) {
      setComparison(null);
      setCompError(null);
      return;
    }
    setCompLoading(true);
    setCompError(null);
    api.backtestCompare(selectedId)
      .then((data) => {
        setComparison(data);
        setCompLoading(false);
      })
      .catch((err) => {
        setCompError(err?.message || 'Failed to load comparison');
        setCompLoading(false);
      });
  }, [selectedId]);

  const handleSelect = useCallback((id: number) => {
    setSelectedId(prev => (prev === id ? null : id));
  }, []);

  const handleCloseComparison = useCallback(() => {
    setSelectedId(null);
  }, []);

  return (
    <PageShell
      title="Simulation Lab"
      subtitle="Live performance, backtest runs, and divergence monitoring"
    >
      {/* Section 1: Live Performance Dashboard */}
      <PerformanceDashboard />

      {/* Section 2: Backtest Runs Table */}
      <BacktestRunsTable runs={runs} selectedId={selectedId} onSelect={handleSelect} />

      {/* Section 3: Live vs Backtest Comparison */}
      {compLoading && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-8">
          <div className="flex items-center justify-center gap-3 text-gray-500">
            <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span className="text-sm">Loading comparison data...</span>
          </div>
        </div>
      )}

      {compError && (
        <div className="bg-surface-1 border border-red-500/30 rounded-xl p-4">
          <div className="flex items-center gap-2 text-red-400 text-sm">
            <svg className="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Failed to load comparison: {compError}</span>
          </div>
        </div>
      )}

      {comparison && !compLoading && (
        <LiveVsBacktestComparison comparison={comparison} onClose={handleCloseComparison} />
      )}
    </PageShell>
  );
}
