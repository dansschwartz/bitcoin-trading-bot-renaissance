import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import MetricCard from '../components/cards/MetricCard';
import ArbDailyPnlChart from '../components/charts/ArbDailyPnlChart';
import ArbDailyPnlBarChart from '../components/charts/ArbDailyPnlBarChart';
import { api } from '../api';
import type { ArbStatus, ArbTrade, ArbSummary, ArbWallet } from '../types';
import { formatCurrency, formatTimestamp, pnlColor, pnlSign, formatUptime } from '../utils/formatters';

const STRATEGY_COLORS: Record<string, string> = {
  cross_exchange: 'bg-accent-blue/20 text-accent-blue',
  triangular: 'bg-purple-400/20 text-purple-400',
  funding_rate: 'bg-accent-green/20 text-accent-green',
  basis_trading: 'bg-orange-400/20 text-orange-400',
  listing_arb: 'bg-pink-400/20 text-pink-400',
  pairs_arb: 'bg-teal-400/20 text-teal-400',
};

const STATUS_COLORS: Record<string, string> = {
  filled: 'bg-accent-green/20 text-accent-green',
  partial: 'bg-accent-yellow/20 text-accent-yellow',
  insufficient_balance: 'bg-gray-600/40 text-gray-400',
  failed: 'bg-accent-red/20 text-accent-red',
  one_sided_fill: 'bg-accent-red/20 text-accent-red',
};

export default function Arbitrage() {
  const [status, setStatus] = useState<ArbStatus | null>(null);
  const [trades, setTrades] = useState<ArbTrade[]>([]);
  const [summary, setSummary] = useState<ArbSummary | null>(null);
  const [wallet, setWallet] = useState<ArbWallet | null>(null);
  const [showSignals, setShowSignals] = useState(false);
  const [velocity, setVelocity] = useState<Record<string, unknown> | null>(null);
  const [edgeDecay, setEdgeDecay] = useState<Record<string, unknown> | null>(null);
  const [allocation, setAllocation] = useState<Record<string, unknown> | null>(null);

  // Helper: format ISO timestamp to relative time string
  const relativeTime = (ts: string | null | undefined): string => {
    if (!ts) return '—';
    try {
      const then = new Date(ts).getTime();
      const now = Date.now();
      const diffMs = now - then;
      if (diffMs < 0) return 'just now';
      const mins = Math.floor(diffMs / 60000);
      if (mins < 1) return 'just now';
      if (mins < 60) return `${mins}m ago`;
      const hrs = Math.floor(mins / 60);
      if (hrs < 24) return `${hrs}h ago`;
      const days = Math.floor(hrs / 24);
      return `${days}d ago`;
    } catch { return '—'; }
  };

  // Helper: format hold duration between two timestamps
  const holdTime = (entry: string | null | undefined, exit: string | null | undefined): string => {
    if (!entry || !exit) return '—';
    try {
      const diffMs = new Date(exit).getTime() - new Date(entry).getTime();
      if (diffMs < 0) return '—';
      const mins = Math.floor(diffMs / 60000);
      if (mins < 60) return `${mins}m`;
      const hrs = Math.floor(mins / 60);
      const remainMins = mins % 60;
      if (hrs < 24) return `${hrs}h ${remainMins}m`;
      const days = Math.floor(hrs / 24);
      return `${days}d ${hrs % 24}h`;
    } catch { return '—'; }
  };
  const [basis, setBasis] = useState<Record<string, unknown> | null>(null);
  const [listing, setListing] = useState<Record<string, unknown> | null>(null);
  const [pairs, setPairs] = useState<Record<string, unknown> | null>(null);
  const [temporal, setTemporal] = useState<Record<string, unknown> | null>(null);
  const [pairExpansion, setPairExpansion] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    const load = () => {
      api.arbStatus().then(setStatus).catch(() => {});
      api.arbTrades(50).then(setTrades).catch(() => {});
      api.arbSummary().then(setSummary).catch(() => {});
      api.arbWallet().then(setWallet).catch(() => {});
      api.arbBasis().then(setBasis).catch(() => {});
      api.arbListing().then(setListing).catch(() => {});
      api.arbPairs().then(setPairs).catch(() => {});
      api.arbTemporal().then(setTemporal).catch(() => {});
      api.arbPairExpansion().then(setPairExpansion).catch(() => {});
    };
    load();
    const id = setInterval(load, 10_000);
    return () => clearInterval(id);
  }, []);

  const isRunning = status?.running ?? false;
  const uptime = status?.uptime_seconds ?? 0;
  const booksConnected = (status?.book_status as Record<string, unknown>)?.tradeable_pairs ?? 0;

  return (
    <PageShell
      title="Arbitrage Engine"
      subtitle="Cross-exchange, triangular, and funding rate arbitrage"
      actions={
        <div className="flex items-center gap-2">
          <span className="px-2 py-1 rounded text-xs font-medium bg-accent-yellow/20 text-accent-yellow">
            PAPER TRADING
          </span>
          <span
            className={`px-2 py-1 rounded text-xs font-medium ${
              isRunning ? 'bg-accent-green/20 text-accent-green' : 'bg-gray-600/40 text-gray-400'
            }`}
          >
            {isRunning ? 'RUNNING' : 'STOPPED'}
          </span>
          {isRunning && (
            <span className="text-xs text-gray-500">
              Uptime: {formatUptime(uptime)} | Books: {String(booksConnected)}
            </span>
          )}
        </div>
      }
    >
      {/* Wallet Section */}
      {wallet && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Arbitrage Wallet</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <div className="text-xs text-gray-500">Initial Balance</div>
              <div className="text-lg font-mono text-gray-200">{formatCurrency(wallet.initial_balance)}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Current Balance</div>
              <div className={`text-lg font-mono ${pnlColor(wallet.total_realized_pnl)}`}>
                {formatCurrency(wallet.current_balance)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Realized P&L</div>
              <div className={`text-lg font-mono ${pnlColor(wallet.total_realized_pnl)}`}>
                {pnlSign(wallet.total_realized_pnl)}{formatCurrency(wallet.total_realized_pnl)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Return</div>
              <div className={`text-lg font-mono ${pnlColor(wallet.return_pct)}`}>
                {pnlSign(wallet.return_pct)}{wallet.return_pct.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Daily P&L Chart */}
      <ArbDailyPnlChart />

      <ArbDailyPnlBarChart />

      {/* Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          title="Total Trades"
          value={summary?.total_trades ?? 0}
          subtitle={`${summary?.filled_trades ?? 0} filled`}
        />
        <MetricCard
          title="Net P&L"
          value={formatCurrency(summary?.total_profit_usd ?? 0)}
          valueColor={pnlColor(summary?.total_profit_usd ?? 0)}
        />
        <MetricCard
          title="Win Rate"
          value={`${((summary?.win_rate ?? 0) * 100).toFixed(1)}%`}
          subtitle={`${summary?.wins ?? 0}W / ${summary?.losses ?? 0}L`}
          valueColor={(summary?.win_rate ?? 0) >= 0.5 ? 'text-accent-green' : 'text-accent-red'}
        />
        <MetricCard
          title="Signals"
          value={summary?.signals_total ?? 0}
          subtitle={`${summary?.signals_approved ?? 0} approved`}
        />
        <MetricCard
          title="Daily P&L"
          value={formatCurrency(summary?.daily_pnl_usd ?? 0)}
          valueColor={pnlColor(summary?.daily_pnl_usd ?? 0)}
        />
        <MetricCard
          title="Filled Rate"
          value={
            summary && summary.total_trades > 0
              ? `${((summary.filled_trades / summary.total_trades) * 100).toFixed(1)}%`
              : '0%'
          }
        />
      </div>

      {/* Strategy Breakdown */}
      {summary?.by_strategy && summary.by_strategy.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {summary.by_strategy.map((s) => (
            <div key={s.strategy} className="bg-surface-1 border border-surface-3 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${STRATEGY_COLORS[s.strategy] || 'bg-surface-3 text-gray-400'}`}>
                  {s.strategy.replace(/_/g, ' ')}
                </span>
              </div>
              <div className="grid grid-cols-4 gap-2 text-xs">
                <div>
                  <div className="text-gray-500">Trades</div>
                  <div className="text-lg font-mono text-gray-200">{s.trades}</div>
                </div>
                <div>
                  <div className="text-gray-500">Fills</div>
                  <div className="text-lg font-mono text-gray-200">{s.fills}</div>
                </div>
                <div>
                  <div className="text-gray-500">Profit</div>
                  <div className={`text-lg font-mono ${pnlColor(s.profit_usd)}`}>
                    {formatCurrency(s.profit_usd)}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Today</div>
                  <div className={`text-lg font-mono ${pnlColor(s.today_pnl_usd ?? 0)}`}>
                    {pnlSign(s.today_pnl_usd ?? 0)}{formatCurrency(s.today_pnl_usd ?? 0)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}



      {/* Temporal Pattern Mining */}
      {temporal && !temporal.error && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300">Temporal Pattern Mining</h3>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-gray-500">
                {temporal.total_profiles ? `${temporal.total_profiles} profiles` : ''}{' '}
                {temporal.total_trades ? `| ${Number(temporal.total_trades).toLocaleString()} trades analyzed` : ''}
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-cyan-400/20 text-cyan-400">
                {temporal.status === 'from_cache' ? 'CACHED' : 'LIVE'}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Best Hours */}
            <div className="bg-surface-2 rounded-lg p-3">
              <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Best Trading Hours (UTC)</h4>
              <div className="space-y-1.5">
                {(temporal.global_best_hours as Array<Record<string, unknown>> ?? []).map((h: Record<string, unknown>, i: number) => {
                  const bias = Number(h.bias ?? 1);
                  const barWidth = Math.min(100, (bias / 2) * 100);
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="w-12 text-gray-400 font-mono">{String(h.hour).padStart(2, '0')}:00</span>
                      <div className="flex-1 h-3 bg-surface-3 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent-green/60 rounded-full"
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                      <span className="w-10 text-right font-mono text-accent-green text-[10px]">{bias.toFixed(2)}x</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Worst Hours */}
            <div className="bg-surface-2 rounded-lg p-3">
              <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Worst Trading Hours (UTC)</h4>
              <div className="space-y-1.5">
                {(temporal.global_worst_hours as Array<Record<string, unknown>> ?? []).map((h: Record<string, unknown>, i: number) => {
                  const bias = Number(h.bias ?? 1);
                  const barWidth = Math.min(100, (bias / 2) * 100);
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="w-12 text-gray-400 font-mono">{String(h.hour).padStart(2, '0')}:00</span>
                      <div className="flex-1 h-3 bg-surface-3 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent-red/60 rounded-full"
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                      <span className="w-10 text-right font-mono text-accent-red text-[10px]">{bias.toFixed(2)}x</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Day of Week */}
            <div className="bg-surface-2 rounded-lg p-3">
              <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Day of Week Bias</h4>
              <div className="space-y-1.5">
                {(temporal.global_by_dow as Array<Record<string, unknown>> ?? []).map((d: Record<string, unknown>, i: number) => {
                  const bias = Number(d.bias ?? 1);
                  const barWidth = Math.min(100, (bias / 2) * 100);
                  const color = bias >= 1.2 ? 'bg-accent-green/60' : bias <= 0.8 ? 'bg-accent-red/60' : 'bg-accent-yellow/60';
                  const textColor = bias >= 1.2 ? 'text-accent-green' : bias <= 0.8 ? 'text-accent-red' : 'text-accent-yellow';
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="w-8 text-gray-400 font-mono">{String(d.day)}</span>
                      <div className="flex-1 h-3 bg-surface-3 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${color} rounded-full`}
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                      <span className={`w-10 text-right font-mono text-[10px] ${textColor}`}>{bias.toFixed(2)}x</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="mt-2 text-[10px] text-gray-600">
            Bias &gt; 1.0 = historically better than average | Bias &lt; 0.4 = trades skipped (temporal filter active)
          </div>
        </div>
      )}

      {/* Pair Expansion */}
      {pairExpansion && pairExpansion.status !== 'not_enabled' && !pairExpansion.error && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300">MEXC Pair Expansion</h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-indigo-400/20 text-indigo-400">
              {(pairExpansion.manager as Record<string, unknown>)?.initialized ? 'ACTIVE' : 'INITIALIZING'}
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">Total Active</div>
              <div className="text-sm font-mono text-gray-200">
                {String((pairExpansion.manager as Record<string, unknown>)?.total_active ?? (pairExpansion.discovery as Record<string, unknown>)?.total_eligible ?? 0)}
              </div>
            </div>
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">Tier 1 (Full)</div>
              <div className="text-sm font-mono text-accent-green">
                {String((pairExpansion.manager as Record<string, unknown>)?.tier_1_count ?? 0)}
              </div>
            </div>
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">Tier 2 (Ticker)</div>
              <div className="text-sm font-mono text-accent-yellow">
                {String((pairExpansion.manager as Record<string, unknown>)?.tier_2_count ?? 0)}
              </div>
            </div>
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">Tier 3 (Tri-only)</div>
              <div className="text-sm font-mono text-gray-400">
                {String((pairExpansion.manager as Record<string, unknown>)?.tier_3_count ?? (pairExpansion.discovery as Record<string, unknown>)?.tier_3_count ?? 0)}
              </div>
            </div>
          </div>

          {/* Tier 1 pairs detail */}
          {Array.isArray((pairExpansion.discovery as Record<string, unknown>)?.tier_1) && (
            <div>
              <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Tier 1 Pairs</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="text-gray-500 border-b border-surface-3">
                      <th className="text-left py-1 px-2">Symbol</th>
                      <th className="text-right py-1 px-2">Score</th>
                      <th className="text-right py-1 px-2">Volume 24h</th>
                      <th className="text-right py-1 px-2">Spread (bps)</th>
                      <th className="text-center py-1 px-2">Binance</th>
                      <th className="text-right py-1 px-2">Tri Paths</th>
                    </tr>
                  </thead>
                  <tbody>
                    {((pairExpansion.discovery as Record<string, unknown>).tier_1 as Array<Record<string, unknown>>).map((p, i) => (
                      <tr key={i} className="border-b border-surface-3/50">
                        <td className="py-1 px-2 text-gray-300">{String(p.symbol)}</td>
                        <td className="py-1 px-2 text-right text-gray-200">{Number(p.score).toFixed(1)}</td>
                        <td className="py-1 px-2 text-right text-gray-400">${Number(p.volume_24h).toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                        <td className="py-1 px-2 text-right text-gray-400">{Number(p.spread_bps).toFixed(1)}</td>
                        <td className="py-1 px-2 text-center">
                          <span className={p.on_binance ? 'text-accent-green' : 'text-gray-500'}>{p.on_binance ? 'Y' : 'N'}</span>
                        </td>
                        <td className="py-1 px-2 text-right text-gray-400">{String(p.triangular_paths ?? 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
      {/* Basis Trading (Spot-Futures Convergence) */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-300">Basis Trading (Spot-Futures)</h3>
          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-orange-400/20 text-orange-400">
            {basis?.observation_mode ? 'OBSERVATION' : 'LIVE'}
          </span>
        </div>
        {basis && (basis as Record<string, unknown>).current_basis && Object.keys((basis as Record<string, unknown>).current_basis as Record<string, unknown>).length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {Object.entries((basis as Record<string, unknown>).current_basis as Record<string, Record<string, unknown>>).map(([sym, data]) => {
              const bps = Number(data.basis_bps ?? 0);
              const dir = String(data.direction ?? 'flat');
              const apr = Number(data.annualized_pct ?? 0);
              const spot = Number(data.spot ?? 0);
              const futures = Number(data.futures ?? 0);
              return (
                <div key={sym} className="bg-surface-2 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-200">{sym}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                      dir === 'contango' ? 'bg-accent-green/20 text-accent-green' :
                      dir === 'backwardation' ? 'bg-accent-red/20 text-accent-red' :
                      'bg-gray-600/40 text-gray-400'
                    }`}>
                      {dir}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-1 text-xs">
                    <div>
                      <div className="text-gray-500">Basis</div>
                      <div className={`font-mono ${bps > 0 ? 'text-accent-green' : bps < 0 ? 'text-accent-red' : 'text-gray-400'}`}>
                        {bps > 0 ? '+' : ''}{bps.toFixed(1)} bps
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-500">APR</div>
                      <div className="font-mono text-gray-300">{apr.toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-gray-500">Spot</div>
                      <div className="font-mono text-gray-400">${spot.toLocaleString(undefined, {maximumFractionDigits: 2})}</div>
                    </div>
                    <div>
                      <div className="text-gray-500">Futures</div>
                      <div className="font-mono text-gray-400">${futures.toLocaleString(undefined, {maximumFractionDigits: 2})}</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-gray-500">No basis data yet — waiting for first scan...</p>
        )}
        {basis != null && Boolean((basis as Record<string, unknown>).stats) && (
          <div className="mt-2 flex gap-4 text-xs text-gray-500">
            <span>Scans: {String(((basis as Record<string, unknown>).stats as Record<string, unknown>)?.scans_completed ?? ((basis as Record<string, unknown>).stats as Record<string, unknown>)?.total_snapshots ?? 0)}</span>
            <span>Opportunities: {String(((basis as Record<string, unknown>).stats as Record<string, unknown>)?.opportunities_found ?? ((basis as Record<string, unknown>).stats as Record<string, unknown>)?.total_opportunities ?? 0)}</span>
          </div>
        )}

        {/* Open Positions */}
        {basis != null && Array.isArray((basis as Record<string, unknown>).open_positions) && ((basis as Record<string, unknown>).open_positions as Array<Record<string, unknown>>).length > 0 && (
          <div className="mt-4">
            <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Open Positions</h4>
            <div className="space-y-2">
              {((basis as Record<string, unknown>).open_positions as Array<Record<string, unknown>>).map((pos, i) => {
                const pnl = Number(pos.unrealized_pnl ?? 0);
                const entryBps = Number(pos.entry_basis_bps ?? 0);
                const curBps = Number(pos.current_basis_bps ?? 0);
                return (
                  <div key={String(pos.position_id ?? i)} className="bg-surface-2 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-gray-200">{String(pos.symbol ?? '—')}</span>
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                          String(pos.direction) === 'buy_basis' ? 'bg-accent-green/20 text-accent-green' : 'bg-accent-red/20 text-accent-red'
                        }`}>
                          {String(pos.direction ?? '—')}
                        </span>
                      </div>
                      <span className={`text-sm font-mono font-medium ${pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                      </span>
                    </div>
                    <div className="grid grid-cols-4 gap-2 text-xs">
                      <div>
                        <div className="text-gray-500">Entry Basis</div>
                        <div className="font-mono text-gray-300">{entryBps.toFixed(1)} bps</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Current Basis</div>
                        <div className="font-mono text-gray-300">{curBps.toFixed(1)} bps</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Size</div>
                        <div className="font-mono text-gray-300">${Number(pos.size_usd ?? 0).toFixed(0)}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Opened</div>
                        <div className="font-mono text-gray-400">{relativeTime(pos.entry_timestamp as string)}</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Recent Trades */}
        {basis != null && Array.isArray((basis as Record<string, unknown>).recent_trades) && ((basis as Record<string, unknown>).recent_trades as Array<Record<string, unknown>>).length > 0 && (
          <div className="mt-4">
            <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Recent Trades</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-surface-3">
                    <th className="text-left py-1 pr-2">Symbol</th>
                    <th className="text-left py-1 pr-2">Direction</th>
                    <th className="text-right py-1 pr-2">Entry→Exit (bps)</th>
                    <th className="text-right py-1 pr-2">P&L</th>
                    <th className="text-right py-1 pr-2">Hold Time</th>
                    <th className="text-left py-1">Exit Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {((basis as Record<string, unknown>).recent_trades as Array<Record<string, unknown>>).slice(0, 10).map((trade, i) => {
                    const pnl = Number(trade.final_pnl ?? 0);
                    return (
                      <tr key={String(trade.position_id ?? i)} className="border-b border-surface-3/50">
                        <td className="py-1.5 pr-2 text-gray-300 font-medium">{String(trade.symbol ?? '—')}</td>
                        <td className="py-1.5 pr-2">
                          <span className={`px-1 py-0.5 rounded text-[10px] ${
                            String(trade.direction) === 'buy_basis' ? 'bg-accent-green/20 text-accent-green' : 'bg-accent-red/20 text-accent-red'
                          }`}>
                            {String(trade.direction ?? '—')}
                          </span>
                        </td>
                        <td className="py-1.5 pr-2 text-right font-mono text-gray-400">
                          {Number(trade.entry_basis_bps ?? 0).toFixed(1)}→{Number(trade.current_basis_bps ?? Number(trade.entry_basis_bps ?? 0)).toFixed(1)}
                        </td>
                        <td className={`py-1.5 pr-2 text-right font-mono font-medium ${pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                          {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                        </td>
                        <td className="py-1.5 pr-2 text-right font-mono text-gray-400">
                          {holdTime(trade.entry_timestamp as string, trade.exit_timestamp as string)}
                        </td>
                        <td className="py-1.5 text-gray-500">{String(trade.exit_reason ?? '—')}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Summary Stats Footer */}
        {basis != null && (
          <div className="mt-4 pt-3 border-t border-surface-3 flex items-center justify-between text-xs">
            <div className="flex gap-4">
              <span className="text-gray-500">Total P&L: <span className={`font-mono font-medium ${
                Number((basis as Record<string, unknown>).total_pnl ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'
              }`}>
                {Number((basis as Record<string, unknown>).total_pnl ?? 0) >= 0 ? '+' : ''}${Number((basis as Record<string, unknown>).total_pnl ?? 0).toFixed(2)}
              </span></span>
              <span className="text-gray-500">Open Positions: <span className="text-gray-300 font-mono">{
                String(((basis as Record<string, unknown>).stats as Record<string, unknown>)?.open_positions ?? 0)
              }</span></span>
              <span className="text-gray-500">Total Trades: <span className="text-gray-300 font-mono">{
                Array.isArray((basis as Record<string, unknown>).recent_trades)
                  ? String(((basis as Record<string, unknown>).recent_trades as unknown[]).length + (Array.isArray((basis as Record<string, unknown>).open_positions) ? ((basis as Record<string, unknown>).open_positions as unknown[]).length : 0))
                  : '0'
              }</span></span>
            </div>
          </div>
        )}
      </div>

      {/* Listing Monitor */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-300">Listing Monitor</h3>
          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-pink-400/20 text-pink-400">
            {listing != null && Boolean((listing as Record<string, unknown>).observation_mode) ? 'OBSERVATION' : 'LIVE'}
          </span>
        </div>

        {/* Monitor stats */}
        {listing != null && Boolean((listing as Record<string, unknown>).monitor_stats) && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">MEXC Symbols</div>
              <div className="text-sm font-mono text-gray-200">
                {String(((listing as Record<string, unknown>).monitor_stats as Record<string, unknown>)?.known_mexc_symbols ?? 0)}
              </div>
            </div>
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">Binance Symbols</div>
              <div className="text-sm font-mono text-gray-200">
                {String(((listing as Record<string, unknown>).monitor_stats as Record<string, unknown>)?.known_binance_symbols ?? 0)}
              </div>
            </div>
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">New Detected</div>
              <div className="text-sm font-mono text-gray-200">
                {String(((listing as Record<string, unknown>).monitor_stats as Record<string, unknown>)?.listings_detected ?? 0)}
              </div>
            </div>
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">Evaluated</div>
              <div className="text-sm font-mono text-gray-200">
                {String((listing as Record<string, unknown>)?.listings_evaluated ?? 0)}
              </div>
            </div>
          </div>
        )}

        {/* Recent listings table */}
        {listing && Array.isArray((listing as Record<string, unknown>).recent_listings) &&
          ((listing as Record<string, unknown>).recent_listings as Array<Record<string, unknown>>).length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="text-left py-1.5 px-2">Symbol</th>
                  <th className="text-left py-1.5 px-2">Detected</th>
                  <th className="text-right py-1.5 px-2">Price</th>
                  <th className="text-center py-1.5 px-2">MEXC First?</th>
                </tr>
              </thead>
              <tbody>
                {((listing as Record<string, unknown>).recent_listings as Array<Record<string, unknown>>).map((evt, i) => (
                  <tr key={i} className="border-b border-surface-3/50">
                    <td className="py-1.5 px-2 text-gray-300">{String(evt.symbol)}</td>
                    <td className="py-1.5 px-2 text-gray-500">{String(evt.detected_at ?? '--').slice(0, 19)}</td>
                    <td className="py-1.5 px-2 text-right text-gray-300">
                      {evt.mexc_initial_price != null ? `$${Number(evt.mexc_initial_price).toFixed(6)}` : '--'}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      <span className={evt.is_first_listing
                        ? 'text-accent-green'
                        : 'text-gray-500'}>
                        {evt.is_first_listing ? 'YES' : 'No'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-xs text-gray-500">No new listings detected yet — monitoring every 60s...</p>
        )}

        {/* Hard limits footer */}
        <div className="mt-2 flex gap-4 text-[10px] text-gray-600">
          <span>Max position: $200</span>
          <span>Max concurrent: 2</span>
          <span>Max hold: 60min</span>
        </div>
      </div>


      {/* Statistical Pairs Trading */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-300">Statistical Pairs Trading</h3>
          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-teal-400/20 text-teal-400">
            {pairs != null && Boolean((pairs as Record<string, unknown>).observation_mode) ? "OBSERVATION" : "LIVE"}
          </span>
        </div>

        {/* Pairs stats */}
        {pairs != null && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">Cointegrated</div>
              <div className="text-sm font-mono text-gray-200">
                {String(((pairs as Record<string, unknown>).pair_states as Array<Record<string, unknown>> ?? []).filter((p: Record<string, unknown>) => p.is_cointegrated).length)}
              </div>
            </div>
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">Scan Cycles</div>
              <div className="text-sm font-mono text-gray-200">
                {String((pairs as Record<string, unknown>).cycle_count ?? 0)}
              </div>
            </div>
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">Signals Logged</div>
              <div className="text-sm font-mono text-gray-200">
                {String((pairs as Record<string, unknown>).total_signals ?? (pairs as Record<string, unknown>).opportunities_detected ?? 0)}
              </div>
            </div>
            <div className="bg-surface-2 rounded-lg p-2">
              <div className="text-[10px] text-gray-500">Opportunities</div>
              <div className="text-sm font-mono text-gray-200">
                {String((pairs as Record<string, unknown>).opportunities_detected ?? 0)}
              </div>
            </div>
          </div>
        )}

        {/* Z-Score Table */}
        {pairs && Array.isArray((pairs as Record<string, unknown>).pair_states) &&
          ((pairs as Record<string, unknown>).pair_states as Array<Record<string, unknown>>).length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="text-left py-1.5 px-2">Pair</th>
                  <th className="text-right py-1.5 px-2">Z-Score</th>
                  <th className="text-right py-1.5 px-2">Half-Life</th>
                  <th className="text-right py-1.5 px-2">Hedge Ratio</th>
                  <th className="text-right py-1.5 px-2">ADF p-val</th>
                  <th className="text-center py-1.5 px-2">Signal</th>
                  <th className="text-center py-1.5 px-2">Cointegrated</th>
                </tr>
              </thead>
              <tbody>
                {((pairs as Record<string, unknown>).pair_states as Array<Record<string, unknown>>).map((ps, i) => {
                  const z = Number(ps.z_score ?? 0);
                  const absZ = Math.abs(z);
                  const zColor = absZ >= 3.5 ? "text-accent-red" :
                                 absZ >= 2.0 ? "text-orange-400" :
                                 absZ >= 0.5 ? "text-accent-yellow" :
                                 "text-gray-500";
                  return (
                    <tr key={i} className="border-b border-surface-3/50">
                      <td className="py-1.5 px-2 text-gray-300">
                        {String(ps.base ?? "")} / {String(ps.quote ?? "")}
                      </td>
                      <td className={"py-1.5 px-2 text-right font-bold " + zColor}>
                        {z >= 0 ? "+" : ""}{z.toFixed(3)}
                      </td>
                      <td className="py-1.5 px-2 text-right text-gray-400">
                        {ps.half_life_bars != null ? Number(ps.half_life_bars).toFixed(1) : "--"}
                      </td>
                      <td className="py-1.5 px-2 text-right text-gray-400">
                        {ps.hedge_ratio != null ? Number(ps.hedge_ratio).toFixed(4) : "--"}
                      </td>
                      <td className="py-1.5 px-2 text-right text-gray-400">
                        {ps.adf_pvalue != null ? Number(ps.adf_pvalue).toFixed(4) : "--"}
                      </td>
                      <td className="py-1.5 px-2 text-center">
                        <span className={
                          String(ps.signal ?? "") === "entry_long" ? "text-accent-green" :
                          String(ps.signal ?? "") === "entry_short" ? "text-accent-red" :
                          String(ps.signal ?? "") === "stop_loss" ? "text-accent-red font-bold" :
                          "text-gray-500"
                        }>
                          {String(ps.signal ?? "none")}
                        </span>
                      </td>
                      <td className="py-1.5 px-2 text-center">
                        <span className={ps.is_cointegrated ? "text-accent-green" : "text-gray-500"}>
                          {ps.is_cointegrated ? "YES" : "No"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-xs text-gray-500">No pair signals yet — waiting for price data accumulation...</p>
        )}

        {/* Z-score color legend */}
        <div className="mt-2 flex gap-4 text-[10px] text-gray-600">
          <span className="text-gray-500">|z| &lt; 0.5 gray</span>
          <span className="text-accent-yellow">|z| 0.5-2.0 yellow</span>
          <span className="text-orange-400">|z| 2.0-3.5 orange (entry)</span>
          <span className="text-accent-red">|z| &gt; 3.5 red (stop)</span>
        </div>
      </div>

      {/* Recent Trades Table */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Recent Trades</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="text-left py-2 px-2">Time</th>
                <th className="text-left py-2 px-2">Strategy</th>
                <th className="text-left py-2 px-2">Symbol</th>
                <th className="text-left py-2 px-2">Buy/Sell</th>
                <th className="text-right py-2 px-2">Size</th>
                <th className="text-right py-2 px-2">Spread (bps)</th>
                <th className="text-right py-2 px-2">Fees</th>
                <th className="text-left py-2 px-2">Status</th>
                <th className="text-right py-2 px-2">P&L</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => {
                const size = t.trade_size_usd || (t.buy_price && t.quantity ? t.buy_price * t.quantity : 0);
                const totalFees = (t.buy_fee ?? 0) + (t.sell_fee ?? 0) + (t.taker_fee_usd ?? 0) + (t.withdrawal_fee_usd ?? 0);
                return (
                <tr key={t.id} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                  <td className="py-2 px-2 text-gray-500">{formatTimestamp(t.timestamp)}</td>
                  <td className="py-2 px-2">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${STRATEGY_COLORS[t.strategy] || 'bg-surface-3 text-gray-400'}`}>
                      {t.strategy}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-gray-300">{t.symbol}</td>
                  <td className="py-2 px-2 text-gray-400">
                    {t.buy_exchange} / {t.sell_exchange}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">
                    {size > 0 ? formatCurrency(size) : '--'}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">
                    {t.net_spread_bps?.toFixed(1) ?? '--'}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-400">
                    {totalFees > 0 ? formatCurrency(totalFees) : '--'}
                  </td>
                  <td className="py-2 px-2">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${STATUS_COLORS[t.status] || 'bg-surface-3 text-gray-400'}`}>
                      {t.status}
                    </span>
                  </td>
                  <td className={`py-2 px-2 text-right ${pnlColor(t.actual_profit_usd)}`}>
                    {t.actual_profit_usd != null
                      ? `${pnlSign(t.actual_profit_usd)}${formatCurrency(t.actual_profit_usd)}`
                      : '--'}
                  </td>
                </tr>
                );
              })}
              {trades.length === 0 && (
                <tr>
                  <td colSpan={9} className="py-4 text-center text-gray-500">
                    No arbitrage trades recorded
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>


      {/* Analytics — Capital Velocity, Edge Decay, Strategy Allocation */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Analytics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

          {/* Capital Velocity */}
          <div className="bg-surface-2 rounded-lg p-3">
            <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Capital Velocity</h4>
            {velocity && (velocity as Record<string, unknown>).strategies ? (
              <div className="space-y-2">
                {Object.entries((velocity as Record<string, unknown>).strategies as Record<string, unknown>).map(([strat, data]) => {
                  const d = data as Record<string, Record<string, unknown>>;
                  const v1h = d?.["1h"]?.velocity;
                  const v24h = d?.["24h"]?.velocity;
                  return (
                    <div key={strat} className="flex justify-between items-center">
                      <span className="text-xs text-gray-400">{strat}</span>
                      <div className="flex gap-3 text-xs font-mono">
                        <span className="text-gray-500">1h: <span className={Number(v1h) > 0 ? 'text-accent-green' : 'text-gray-400'}>{Number(v1h || 0).toFixed(4)}</span></span>
                        <span className="text-gray-500">24h: <span className={Number(v24h) > 0 ? 'text-accent-green' : 'text-gray-400'}>{Number(v24h || 0).toFixed(4)}</span></span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-gray-500">No velocity data yet</p>
            )}
          </div>

          {/* Edge Decay */}
          <div className="bg-surface-2 rounded-lg p-3">
            <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Edge Decay</h4>
            {edgeDecay && (edgeDecay as Record<string, unknown>).strategies ? (
              <div className="space-y-2">
                {Object.entries((edgeDecay as Record<string, unknown>).strategies as Record<string, unknown>).map(([strat, data]) => {
                  const d = data as Record<string, unknown>;
                  const health = String(d?.health ?? "unknown");
                  const slope = Number(d?.slope ?? 0);
                  const healthColor = health === "healthy" ? "bg-green-400/20 text-green-400"
                    : health === "warning" ? "bg-yellow-400/20 text-yellow-400"
                    : health === "critical" ? "bg-orange-400/20 text-orange-400"
                    : health === "dead" ? "bg-red-400/20 text-red-400"
                    : "bg-gray-400/20 text-gray-400";
                  return (
                    <div key={strat} className="flex justify-between items-center">
                      <span className="text-xs text-gray-400">{strat}</span>
                      <div className="flex gap-2 items-center">
                        <span className="text-xs font-mono text-gray-500">{slope >= 0 ? '+' : ''}{slope.toFixed(4)}</span>
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${healthColor}`}>
                          {health.toUpperCase()}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-gray-500">No edge decay data yet</p>
            )}
          </div>

          {/* Strategy Allocation */}
          <div className="bg-surface-2 rounded-lg p-3">
            <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Strategy Allocation</h4>
            {allocation && (allocation as Record<string, unknown>).target_allocation ? (
              <div className="space-y-2">
                {allocation && Boolean((allocation as Record<string, unknown>).observation_mode) && (
                  <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-amber-400/20 text-amber-400 mb-1 inline-block">
                    OBSERVATION MODE
                  </span>
                )}
                {Object.entries((allocation as Record<string, unknown>).target_allocation as Record<string, unknown>).map(([strat, pct]) => {
                  const current = ((allocation as Record<string, unknown>).current_allocation as Record<string, unknown>)?.[strat];
                  const pctNum = Number(pct) * 100;
                  const curNum = Number(current ?? pct) * 100;
                  const barColor = strat === "cross_exchange" ? "bg-blue-500"
                    : strat === "triangular" ? "bg-purple-500"
                    : "bg-cyan-500";
                  return (
                    <div key={strat}>
                      <div className="flex justify-between text-xs mb-0.5">
                        <span className="text-gray-400">{strat}</span>
                        <span className="font-mono text-gray-300">{pctNum.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-surface-3 rounded-full h-1.5">
                        <div className={`${barColor} h-1.5 rounded-full`} style={{ width: `${Math.min(100, pctNum)}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-gray-500">Allocator has not run yet</p>
            )}
          </div>

        </div>
      </div>

      {/* Signals (collapsible) */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <button
          onClick={() => setShowSignals(!showSignals)}
          className="flex items-center gap-2 text-sm font-medium text-gray-300 hover:text-gray-100"
        >
          <svg className={`w-4 h-4 transition-transform ${showSignals ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          Recent Signals
        </button>
        {showSignals && <SignalsTable />}
      </div>
    </PageShell>
  );
}

function SignalsTable() {
  const [signals, setSignals] = useState<Array<Record<string, unknown>>>([]);

  useEffect(() => {
    api.arbSignals(100).then((d) => setSignals(d as unknown as Array<Record<string, unknown>>)).catch(() => {});
  }, []);

  if (signals.length === 0) {
    return <p className="text-xs text-gray-500 mt-2">No signals recorded yet.</p>;
  }

  return (
    <div className="overflow-x-auto mt-3">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-gray-500 border-b border-surface-3">
            <th className="text-left py-2 px-2">Time</th>
            <th className="text-left py-2 px-2">Strategy</th>
            <th className="text-left py-2 px-2">Symbol</th>
            <th className="text-right py-2 px-2">Gross (bps)</th>
            <th className="text-right py-2 px-2">Net (bps)</th>
            <th className="text-center py-2 px-2">Approved</th>
            <th className="text-center py-2 px-2">Executed</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s, i) => (
            <tr key={i} className="border-b border-surface-3/50">
              <td className="py-2 px-2 text-gray-500">{formatTimestamp(String(s.timestamp ?? ''))}</td>
              <td className="py-2 px-2">
                <span className={`px-1.5 py-0.5 rounded text-[10px] ${STRATEGY_COLORS[String(s.strategy ?? '')] || 'bg-surface-3 text-gray-400'}`}>
                  {String(s.strategy ?? '')}
                </span>
              </td>
              <td className="py-2 px-2 text-gray-300">{String(s.symbol ?? '')}</td>
              <td className="py-2 px-2 text-right text-gray-300">
                {typeof s.gross_spread_bps === 'number' ? s.gross_spread_bps.toFixed(1) : '--'}
              </td>
              <td className="py-2 px-2 text-right text-gray-300">
                {typeof s.net_spread_bps === 'number' ? s.net_spread_bps.toFixed(1) : '--'}
              </td>
              <td className="py-2 px-2 text-center">
                <span className={s.approved ? 'text-accent-green' : 'text-gray-500'}>
                  {s.approved ? 'Y' : 'N'}
                </span>
              </td>
              <td className="py-2 px-2 text-center">
                <span className={s.executed ? 'text-accent-green' : 'text-gray-500'}>
                  {s.executed ? 'Y' : 'N'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
