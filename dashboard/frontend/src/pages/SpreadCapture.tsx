import { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, Cell, Legend,
} from 'recharts';
import { api } from '../api';

// ─── Types ───────────────────────────────────────────────────
interface WalletData {
  usdc_balance: number;
  last_updated: number;
  rpc: string;
  error: string;
  starting_balance_today: number;
  first_balance_ever: number;
  actual_daily_pnl: number;
  actual_total_pnl: number;
  bot_estimated_pnl: number;
  tracking_error: number;
  wallet_address: string;
}

interface Summary {
  total_windows: number;
  wins: number;
  losses: number;
  win_rate: number;
  arb_windows: number;
  arb_rate: number;
  total_pnl: number;
  avg_pair_cost: number;
  avg_pnl_per_window: number;
  active_positions: number;
  daily_pnl: number;
  daily_windows: number;
  total_deployed: number;
}

interface ActivePosition {
  slug: string;
  asset: string;
  timeframe: string;
  yes_shares: number;
  yes_avg_price: number;
  no_shares: number;
  no_avg_price: number;
  pair_cost: number;
  total_cost: number;
  is_arb: boolean;
  guaranteed_profit: number;
  phase: number;
}

interface Window {
  id: number;
  asset: string;
  timeframe: string;
  slug: string;
  pair_cost: number;
  total_deployed: number;
  guaranteed_profit: number;
  is_arb: number;
  resolution: string;
  pnl: number;
  pnl_pct: number;
  max_phase: number;
  phase2_triggered: number;
  resolved_at: string;
  yes_shares: number;
  no_shares: number;
}

interface PnlChartData {
  bot_series: { ts: string; pnl: number; cumulative: number; pair_cost: number; asset: string; timeframe: string }[];
  wallet_series: { ts: string; wallet_pnl: number; wallet_balance: number }[];
}

interface AssetPerf {
  asset: string;
  total: number;
  wins: number;
  total_pnl: number;
  avg_pnl: number;
  avg_pair_cost: number;
  arb_count: number;
  phase2_count: number;
}

interface RtdsAsset {
  binance: number | null;
  chainlink: number | null;
  spread: number | null;
}

interface RtdsStatus {
  connected: boolean;
  assets: Record<string, RtdsAsset>;
}

interface PairCostBucket {
  bucket: string;
  count: number;
  avg_pnl: number;
}

interface Fill {
  id: number;
  window_slug: string;
  side: string;
  price: number;
  shares: number;
  amount_usd: number;
  phase: number;
  timestamp: number;
}

// ─── Merged chart data ──────────────────────────────────────
interface MergedPoint {
  ts: string;
  cumulative?: number;
  wallet_pnl?: number;
}

function mergeChartData(pnlChart: PnlChartData | null): MergedPoint[] {
  if (!pnlChart) return [];
  const map = new Map<string, MergedPoint>();

  for (const p of pnlChart.bot_series) {
    const key = p.ts?.slice(0, 16) || '';
    if (!key) continue;
    const existing = map.get(key) || { ts: key };
    existing.cumulative = p.cumulative;
    map.set(key, existing);
  }

  for (const w of pnlChart.wallet_series) {
    const key = w.ts?.slice(0, 16) || '';
    if (!key) continue;
    const existing = map.get(key) || { ts: key };
    existing.wallet_pnl = w.wallet_pnl;
    map.set(key, existing);
  }

  return Array.from(map.values()).sort((a, b) => a.ts.localeCompare(b.ts));
}

// ─── Component ───────────────────────────────────────────────
export default function SpreadCapture() {
  const [wallet, setWallet] = useState<WalletData | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [active, setActive] = useState<ActivePosition[]>([]);
  const [history, setHistory] = useState<Window[]>([]);
  const [pnlChart, setPnlChart] = useState<PnlChartData | null>(null);
  const [byAsset, setByAsset] = useState<AssetPerf[]>([]);
  const [rtds, setRtds] = useState<RtdsStatus | null>(null);
  const [pairDist, setPairDist] = useState<PairCostBucket[]>([]);
  const [fills, setFills] = useState<Fill[]>([]);
  const [tab, setTab] = useState<'overview' | 'fills' | 'history'>('overview');

  const refresh = useCallback(async () => {
    try {
      const [w, s, a, h, p, ba, r, pd, f] = await Promise.all([
        api.scWallet(),
        api.scSummary(),
        api.scActive(),
        api.scHistory(50),
        api.scPnlChart(),
        api.scByAsset(),
        api.scRtds(),
        api.scPairCostDist(),
        api.scFills(100),
      ]);
      setWallet(w as unknown as WalletData);
      setSummary(s as unknown as Summary);
      setActive(a as unknown as ActivePosition[]);
      setHistory(h as unknown as Window[]);
      setPnlChart(p as unknown as PnlChartData);
      setByAsset(ba as unknown as AssetPerf[]);
      setRtds(r as unknown as RtdsStatus);
      setPairDist(pd as unknown as PairCostBucket[]);
      setFills(f as unknown as Fill[]);
    } catch (e) {
      console.error('Spread capture fetch error:', e);
    }
  }, []);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 5000);
    return () => clearInterval(iv);
  }, [refresh]);

  const pnlColor = (v: number) => v >= 0 ? 'text-green-400' : 'text-red-400';
  const merged = mergeChartData(pnlChart);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Spread Capture</h1>
          <p className="text-gray-400 text-sm mt-1">
            0x8dxd Strategy — Buy favorite + accumulate underdog = pair cost &lt; $1.00
          </p>
        </div>
        <div className="flex items-center gap-2">
          {rtds?.connected ? (
            <span className="px-2 py-1 rounded text-xs bg-green-500/20 text-green-400">RTDS LIVE</span>
          ) : (
            <span className="px-2 py-1 rounded text-xs bg-red-500/20 text-red-400">RTDS OFFLINE</span>
          )}
          <span className="px-2 py-1 rounded text-xs bg-blue-500/20 text-blue-400">
            {summary?.active_positions ?? 0} Active
          </span>
        </div>
      </div>

      {/* ══════════ GROUND TRUTH: Wallet Balance Cards ══════════ */}
      {wallet && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {/* Wallet Balance */}
          <div className="bg-surface-1 rounded-xl p-4 border border-white/10">
            <div className="text-xs text-gray-400 mb-1 flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${wallet.error ? 'bg-red-400' : 'bg-green-400'}`} />
              Wallet Balance
              <span className="text-gray-600 ml-auto">from Polygon</span>
            </div>
            <div className="text-3xl font-mono font-bold text-white">
              ${wallet.usdc_balance.toFixed(2)}
            </div>
            <div className="text-xs text-gray-500 mt-1 font-mono truncate" title={wallet.wallet_address}>
              {wallet.wallet_address.slice(0, 10)}...{wallet.wallet_address.slice(-6)}
            </div>
          </div>

          {/* Actual P&L */}
          <div className="bg-surface-1 rounded-xl p-4 border border-white/10">
            <div className="text-xs text-gray-400 mb-1">Actual P&L (wallet)</div>
            <div className={`text-3xl font-mono font-bold ${pnlColor(wallet.actual_total_pnl)}`}>
              ${wallet.actual_total_pnl >= 0 ? '+' : ''}{wallet.actual_total_pnl.toFixed(2)}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Today: <span className={pnlColor(wallet.actual_daily_pnl)}>
                ${wallet.actual_daily_pnl >= 0 ? '+' : ''}{wallet.actual_daily_pnl.toFixed(2)}
              </span>
              <span className="mx-2 text-gray-600">|</span>
              Start: ${wallet.first_balance_ever.toFixed(2)}
            </div>
          </div>

          {/* Tracking Error */}
          <div className="bg-surface-1 rounded-xl p-4 border border-white/10">
            <div className="text-xs text-gray-400 mb-1">Tracking Error</div>
            <div className={`text-3xl font-mono font-bold ${
              Math.abs(wallet.tracking_error) < 10 ? 'text-green-400' :
              Math.abs(wallet.tracking_error) < 50 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              ${Math.abs(wallet.tracking_error).toFixed(2)}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Bot says: <span className="text-gray-400">${wallet.bot_estimated_pnl.toFixed(2)}</span>
              <span className="mx-2 text-gray-600">|</span>
              {wallet.tracking_error > 0 ? 'Bot overstates' : 'Bot understates'}
            </div>
          </div>
        </div>
      )}

      {/* Bot estimated P&L cards (secondary) */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <Card label="Est. P&L (bot)" value={`$${summary.total_pnl.toFixed(2)}`}
                className={`${pnlColor(summary.total_pnl)} opacity-60`} />
          <Card label="Windows" value={String(summary.total_windows)} />
          <Card label="Win Rate" value={`${summary.win_rate}%`}
                className={summary.win_rate >= 50 ? 'text-green-400' : 'text-yellow-400'} />
          <Card label="Avg Pair Cost" value={`$${summary.avg_pair_cost.toFixed(3)}`}
                className={summary.avg_pair_cost < 1.0 ? 'text-green-400' : 'text-yellow-400'} />
          <Card label="Active" value={String(summary.active_positions)} />
          <Card label="Arb Rate" value={`${summary.arb_rate}%`}
                className={summary.arb_rate >= 50 ? 'text-green-400' : 'text-gray-400'} />
        </div>
      )}

      {/* Tab Selector */}
      <div className="flex gap-1 bg-surface-1 rounded-lg p-1 w-fit">
        {(['overview', 'fills', 'history'] as const).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-1.5 rounded text-sm capitalize transition-colors ${
              tab === t ? 'bg-accent-blue text-white' : 'text-gray-400 hover:text-gray-200'
            }`}>
            {t}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          {/* ══════════ DUAL P&L CHART ══════════ */}
          {merged.length > 0 && (
            <div className="bg-surface-1 rounded-xl p-4">
              <h2 className="text-sm font-semibold text-gray-300 mb-1">P&L Comparison</h2>
              <p className="text-xs text-gray-500 mb-3">
                White = actual wallet change (ground truth) | Green = bot estimated (may be overstated)
              </p>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={merged}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                  <XAxis dataKey="ts" tick={{ fontSize: 10 }} stroke="#666"
                    tickFormatter={(v: string) => v ? v.slice(11, 16) : ''} />
                  <YAxis tick={{ fontSize: 10 }} stroke="#666"
                    tickFormatter={(v: number) => `$${v.toFixed(0)}`} />
                  <Tooltip
                    contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                    labelStyle={{ color: '#999' }}
                    formatter={(v: number, name: string) => [`$${v.toFixed(2)}`, name]}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="wallet_pnl" stroke="#ffffff"
                    dot={false} strokeWidth={2.5} name="Actual (wallet)"
                    connectNulls />
                  <Line type="monotone" dataKey="cumulative" stroke="#4ade80"
                    dot={false} strokeWidth={1.5} name="Estimated (bot)"
                    strokeDasharray="4 2" connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Active Positions */}
          {active.length > 0 && (
            <div className="bg-surface-1 rounded-xl p-4">
              <h2 className="text-sm font-semibold text-gray-300 mb-3">
                Active Positions ({active.length})
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-500 text-xs border-b border-surface-3">
                      <th className="text-left py-2 px-2">Asset</th>
                      <th className="text-left py-2 px-2">TF</th>
                      <th className="text-right py-2 px-2">YES</th>
                      <th className="text-right py-2 px-2">NO</th>
                      <th className="text-right py-2 px-2">Pair Cost</th>
                      <th className="text-right py-2 px-2">Deployed</th>
                      <th className="text-right py-2 px-2">Guar. Profit</th>
                      <th className="text-center py-2 px-2">Phase</th>
                      <th className="text-center py-2 px-2">Arb?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {active.map((p, i) => (
                      <tr key={i} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                        <td className="py-2 px-2 font-mono font-semibold">{p.asset}</td>
                        <td className="py-2 px-2 text-gray-400">{p.timeframe}</td>
                        <td className="py-2 px-2 text-right font-mono">
                          {p.yes_shares.toFixed(1)} @ ${p.yes_avg_price.toFixed(3)}
                        </td>
                        <td className="py-2 px-2 text-right font-mono">
                          {p.no_shares.toFixed(1)} @ ${p.no_avg_price.toFixed(3)}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono font-semibold ${
                          p.pair_cost < 1.0 ? 'text-green-400' : 'text-yellow-400'
                        }`}>${p.pair_cost.toFixed(3)}</td>
                        <td className="py-2 px-2 text-right font-mono">${p.total_cost.toFixed(2)}</td>
                        <td className={`py-2 px-2 text-right font-mono ${pnlColor(p.guaranteed_profit)}`}>
                          ${p.guaranteed_profit.toFixed(2)}
                        </td>
                        <td className="py-2 px-2 text-center">
                          <span className={`px-2 py-0.5 rounded text-xs ${
                            p.phase >= 2 ? 'bg-purple-500/20 text-purple-400' : 'bg-blue-500/20 text-blue-400'
                          }`}>P{p.phase}</span>
                        </td>
                        <td className="py-2 px-2 text-center">
                          {p.is_arb ? <span className="text-green-400">&#10003;</span> : <span className="text-gray-600">-</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Asset Breakdown + Pair Cost Distribution */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {byAsset.length > 0 && (
              <div className="bg-surface-1 rounded-xl p-4">
                <h2 className="text-sm font-semibold text-gray-300 mb-3">By Asset</h2>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-500 text-xs border-b border-surface-3">
                      <th className="text-left py-2">Asset</th>
                      <th className="text-right py-2">Windows</th>
                      <th className="text-right py-2">Win%</th>
                      <th className="text-right py-2">P&L</th>
                      <th className="text-right py-2">Avg Cost</th>
                      <th className="text-right py-2">Arb%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {byAsset.map((a, i) => (
                      <tr key={i} className="border-b border-surface-3/50">
                        <td className="py-1.5 font-mono font-semibold">{a.asset}</td>
                        <td className="py-1.5 text-right">{a.total}</td>
                        <td className="py-1.5 text-right">
                          {a.total > 0 ? ((a.wins / a.total) * 100).toFixed(0) : 0}%
                        </td>
                        <td className={`py-1.5 text-right font-mono ${pnlColor(a.total_pnl)}`}>
                          ${a.total_pnl.toFixed(3)}
                        </td>
                        <td className={`py-1.5 text-right font-mono ${
                          a.avg_pair_cost < 1.0 ? 'text-green-400' : 'text-yellow-400'
                        }`}>${a.avg_pair_cost.toFixed(3)}</td>
                        <td className="py-1.5 text-right">
                          {a.total > 0 ? ((a.arb_count / a.total) * 100).toFixed(0) : 0}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {pairDist.length > 0 && (
              <div className="bg-surface-1 rounded-xl p-4">
                <h2 className="text-sm font-semibold text-gray-300 mb-3">Pair Cost Distribution</h2>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={pairDist}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                    <XAxis dataKey="bucket" tick={{ fontSize: 10 }} stroke="#666" />
                    <YAxis tick={{ fontSize: 10 }} stroke="#666" />
                    <Tooltip
                      contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                      formatter={(v: number, name: string) => [
                        name === 'avg_pnl' ? `$${v.toFixed(3)}` : v, name
                      ]}
                    />
                    <Bar dataKey="count" name="Windows" radius={[4, 4, 0, 0]}>
                      {pairDist.map((entry, i) => (
                        <Cell key={i} fill={
                          entry.bucket.includes('<') || entry.bucket.includes('0.9') || entry.bucket === '0.95-1.00'
                            ? '#4ade80' : entry.bucket.includes('1.0') ? '#facc15' : '#ef4444'
                        } />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* RTDS Oracle Feeds */}
          {rtds && rtds.assets && Object.keys(rtds.assets).length > 0 && (
            <div className="bg-surface-1 rounded-xl p-4">
              <h2 className="text-sm font-semibold text-gray-300 mb-3">Oracle Feeds (RTDS)</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                {Object.entries(rtds.assets).map(([asset, data]) => (
                  <div key={asset} className="bg-surface-2 rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-400 mb-1">{asset}</div>
                    <div className="font-mono text-sm">
                      {data.binance ? `$${data.binance.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : '-'}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      CL: {data.chainlink ? `$${data.chainlink.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : '-'}
                    </div>
                    {data.spread !== null && data.spread !== undefined && (
                      <div className={`text-xs mt-1 font-mono ${Math.abs(data.spread) < 1 ? 'text-gray-500' : 'text-yellow-400'}`}>
                        {data.spread >= 0 ? '+' : ''}{data.spread.toFixed(2)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {tab === 'fills' && (
        <div className="bg-surface-1 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Recent Fills ({fills.length})</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs border-b border-surface-3">
                  <th className="text-left py-2 px-2">Time</th>
                  <th className="text-left py-2 px-2">Window</th>
                  <th className="text-center py-2 px-2">Side</th>
                  <th className="text-right py-2 px-2">Price</th>
                  <th className="text-right py-2 px-2">Shares</th>
                  <th className="text-right py-2 px-2">Amount</th>
                  <th className="text-center py-2 px-2">Phase</th>
                </tr>
              </thead>
              <tbody>
                {fills.map((f, i) => (
                  <tr key={i} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                    <td className="py-1.5 px-2 text-xs text-gray-400 font-mono">
                      {f.timestamp ? new Date(f.timestamp * 1000).toLocaleTimeString() : '-'}
                    </td>
                    <td className="py-1.5 px-2 font-mono text-xs truncate max-w-[200px]" title={f.window_slug}>
                      {f.window_slug?.split('-').slice(0, 3).join('-')}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        f.side === 'YES' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                      }`}>{f.side}</span>
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono">${f.price.toFixed(3)}</td>
                    <td className="py-1.5 px-2 text-right font-mono">{f.shares.toFixed(1)}</td>
                    <td className="py-1.5 px-2 text-right font-mono">${f.amount_usd.toFixed(2)}</td>
                    <td className="py-1.5 px-2 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        f.phase === 1 ? 'bg-blue-500/20 text-blue-400' : 'bg-purple-500/20 text-purple-400'
                      }`}>P{f.phase}</span>
                    </td>
                  </tr>
                ))}
                {fills.length === 0 && (
                  <tr><td colSpan={7} className="py-8 text-center text-gray-500">No fills yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'history' && (
        <div className="bg-surface-1 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Resolved Windows ({history.length})</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs border-b border-surface-3">
                  <th className="text-left py-2 px-2">Time</th>
                  <th className="text-left py-2 px-2">Asset</th>
                  <th className="text-left py-2 px-2">TF</th>
                  <th className="text-center py-2 px-2">Result</th>
                  <th className="text-right py-2 px-2">Pair Cost</th>
                  <th className="text-right py-2 px-2">Deployed</th>
                  <th className="text-right py-2 px-2">P&L</th>
                  <th className="text-right py-2 px-2">P&L %</th>
                  <th className="text-center py-2 px-2">Phase</th>
                  <th className="text-center py-2 px-2">Arb?</th>
                </tr>
              </thead>
              <tbody>
                {history.map((w, i) => (
                  <tr key={i} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                    <td className="py-1.5 px-2 text-xs text-gray-400 font-mono">
                      {w.resolved_at ? w.resolved_at.slice(11, 19) : '-'}
                    </td>
                    <td className="py-1.5 px-2 font-mono font-semibold">{w.asset}</td>
                    <td className="py-1.5 px-2 text-gray-400">{w.timeframe}</td>
                    <td className="py-1.5 px-2 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        w.resolution === 'UP' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                      }`}>{w.resolution || '?'}</span>
                    </td>
                    <td className={`py-1.5 px-2 text-right font-mono ${
                      w.pair_cost < 1.0 ? 'text-green-400' : 'text-yellow-400'
                    }`}>${w.pair_cost?.toFixed(3) ?? '-'}</td>
                    <td className="py-1.5 px-2 text-right font-mono">${w.total_deployed?.toFixed(2) ?? '-'}</td>
                    <td className={`py-1.5 px-2 text-right font-mono font-semibold ${pnlColor(w.pnl)}`}>
                      ${w.pnl?.toFixed(3) ?? '-'}
                    </td>
                    <td className={`py-1.5 px-2 text-right font-mono ${pnlColor(w.pnl_pct)}`}>
                      {w.pnl_pct?.toFixed(1) ?? '-'}%
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        w.max_phase >= 2 ? 'bg-purple-500/20 text-purple-400' : 'bg-blue-500/20 text-blue-400'
                      }`}>P{w.max_phase}</span>
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      {w.is_arb ? <span className="text-green-400">&#10003;</span> : <span className="text-gray-600">-</span>}
                    </td>
                  </tr>
                ))}
                {history.length === 0 && (
                  <tr><td colSpan={10} className="py-8 text-center text-gray-500">No resolved windows yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!summary && !wallet && (
        <div className="bg-surface-1 rounded-xl p-12 text-center text-gray-500">
          Loading spread capture data...
        </div>
      )}
    </div>
  );
}

function Card({ label, value, className = '' }: { label: string; value: string; className?: string }) {
  return (
    <div className="bg-surface-1 rounded-xl p-3">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-lg font-mono font-semibold ${className}`}>{value}</div>
    </div>
  );
}
