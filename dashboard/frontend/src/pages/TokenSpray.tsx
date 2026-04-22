import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import { api } from '../api';
import { Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Line, ComposedChart } from 'recharts';

interface SprayStatus {
  active: boolean;
  observation_mode: boolean;
  spray_interval_seconds: number;
  total_sprayed: number;
  total_open: number;
  max_open: number;
  today_pnl_usd: number;
  today_tokens: number;
  budget: {
    deployed_usd: number;
    total_capital: number;
    deployed_pct: number;
  };
}

interface HourlyPnl {
  hour: string;
  tokens: number;
  winners: number;
  pnl_usd: number;
  avg_pnl_bps: number;
  win_rate: number;
}

interface AbResult {
  rule: string;
  pair: string;
  trades: number;
  win_rate: number;
  total_pnl_bps: number;
  avg_pnl_bps: number;
}

interface PairEcon {
  pair: string;
  direction_rule: string;
  total_tokens: number;
  winners: number;
  win_rate: number;
  avg_pnl_bps: number;
  total_pnl_usd: number;
  avg_win_bps: number | null;
  avg_loss_bps: number | null;
  avg_hold_sec: number | null;
  current_ev: number;
  current_spread: number;
  tradeable: boolean;
}

interface TokenEvent {
  timestamp: string;
  pair: string;
  direction: string;
  token_size_usd: number;
  entry_price: number;
  direction_rule: string;
  vol_regime: string;
  exit_pnl_bps: number | null;
  exit_pnl_usd: number | null;
  exit_reason: string | null;
  hold_time_seconds: number | null;
  observation_mode: number;
}

interface VolRegime {
  regime: string;
  predicted_bps: number;
  tradeable: boolean;
}

const VOL_COLORS: Record<string, string> = {
  active: 'bg-accent-green/20 text-accent-green',
  explosive: 'bg-accent-yellow/20 text-accent-yellow',
  normal: 'bg-blue-500/20 text-blue-400',
  dead_zone: 'bg-accent-red/20 text-accent-red',
};

function shortenExitReason(reason: string | null): string {
  if (!reason) return '--';
  return reason
    .replace('EDGE_CONSUMED_EARLY', 'EDGE')
    .replace('EDGE_CONSUMED', 'EDGE')
    .replace('STOP_LOSS_EARLY', 'STOP')
    .replace('STOP_LOSS', 'STOP')
    .replace('DIRECTION_REVERSAL', 'REVERSAL')
    .replace('VOLATILITY_SPIKE', 'VOL_SPIKE')
    .replace('sub_bar:', '');
}

function formatTime(ts: string | null): string {
  if (!ts) return '--';
  try {
    const parts = ts.split(' ');
    return parts[1]?.substring(0, 5) || ts.substring(11, 16) || '--';
  } catch { return '--'; }
}

export default function TokenSpray() {
  const [status, setStatus] = useState<SprayStatus | null>(null);
  const [hourly, setHourly] = useState<HourlyPnl[]>([]);
  const [abResults, setAbResults] = useState<Record<string, AbResult>>({});
  const [pairEcon, setPairEcon] = useState<PairEcon[]>([]);
  const [feed, setFeed] = useState<TokenEvent[]>([]);
  const [volRegimes, setVolRegimes] = useState<Record<string, VolRegime>>({});

  const refresh = () => {
    api.sprayStatus().then(d => setStatus(d as unknown as SprayStatus)).catch(() => {});
    api.sprayHourlyPnl().then(d => setHourly([...(d as unknown as HourlyPnl[])].reverse())).catch(() => {});
    api.sprayAbTest().then(d => setAbResults((d as Record<string, unknown>).ab_results as Record<string, AbResult> || {})).catch(() => {});
    api.sprayPairEconomics().then(d => setPairEcon(d as unknown as PairEcon[])).catch(() => {});
    api.sprayLiveFeed().then(d => setFeed(d as unknown as TokenEvent[])).catch(() => {});
    api.sprayVolatility().then(d => setVolRegimes(d as unknown as Record<string, VolRegime>)).catch(() => {});
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  // Aggregate A/B results by rule
  const byRule: Record<string, { trades: number; wins: number; pnl: number }> = {};
  for (const data of Object.values(abResults)) {
    const rule = data.rule || 'unknown';
    if (!byRule[rule]) byRule[rule] = { trades: 0, wins: 0, pnl: 0 };
    byRule[rule].trades += data.trades;
    byRule[rule].wins += Math.round((data.win_rate || 0) * data.trades);
    byRule[rule].pnl += data.total_pnl_bps || 0;
  }
  const abSorted = Object.entries(byRule)
    .sort((a, b) => {
      const wrA = a[1].trades > 0 ? a[1].wins / a[1].trades : 0;
      const wrB = b[1].trades > 0 ? b[1].wins / b[1].trades : 0;
      return wrB - wrA;
    });
  return (
    <PageShell title="Token Spray" subtitle="Spray status, A/B direction test, pair economics, live feed">
      {/* Status Bar */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-3">
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <span>Mode: <span className={`inline-block px-2 py-0.5 rounded text-xs font-mono ${
            status?.observation_mode ? 'bg-accent-yellow/20 text-accent-yellow' : 'bg-accent-green/20 text-accent-green'
          }`}>{status?.observation_mode ? 'OBSERVATION' : 'ACTIVE'}</span></span>
          <span className="text-gray-400">Interval: <strong className="text-gray-200">{status?.spray_interval_seconds ?? 5}s</strong></span>
          <span className="text-gray-400">Open: <strong className="text-gray-200">{status?.total_open ?? 0} / {status?.max_open ?? 200}</strong></span>
          <span className="text-gray-400">Sprayed: <strong className="text-gray-200">{(status?.total_sprayed ?? 0).toLocaleString()}</strong></span>
          <span className="text-gray-400">Deployed: <strong className="text-gray-200">
            ${(status?.budget?.deployed_usd ?? 0).toFixed(0)} / ${((status?.budget?.total_capital ?? 5000) * 0.8).toFixed(0)}
          </strong> <span className="text-gray-500">({(status?.budget?.deployed_pct ?? 0).toFixed(0)}%)</span></span>
          <span className="text-gray-400">Today: <strong className={`${(status?.today_pnl_usd ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {(status?.today_pnl_usd ?? 0) >= 0 ? '+' : ''}${(status?.today_pnl_usd ?? 0).toFixed(2)}
          </strong></span>
          <div className="w-24 h-2 bg-surface-3 rounded-full overflow-hidden">
            <div className={`h-full rounded-full ${(status?.budget?.deployed_pct ?? 0) > 70 ? 'bg-accent-yellow' : 'bg-accent-green'}`}
                 style={{ width: `${Math.min(status?.budget?.deployed_pct ?? 0, 100)}%` }} />
          </div>
        </div>
      </div>

      {/* Volatility Regime Strip */}
      {Object.keys(volRegimes).length > 0 && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-3">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-gray-500 font-medium mr-1">Vol Regimes:</span>
            {Object.entries(volRegimes).map(([pair, v]) => (
              <span key={pair} className={`px-2 py-0.5 rounded font-mono ${VOL_COLORS[v.regime] || 'bg-surface-2 text-gray-400'}`}>
                {pair.replace('-USD', '')}: {v.regime} {v.predicted_bps}bp
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Hourly P&L Chart */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Hourly P&L + Win Rate</h3>
        {hourly.length === 0 ? (
          <div className="text-sm text-gray-600 py-8 text-center">No hourly data yet — waiting for closed tokens</div>
        ) : (
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={hourly} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                <XAxis dataKey="hour" tick={{ fontSize: 10, fill: '#6b7280' }}
                       tickFormatter={h => h?.split(' ')[1] || h} />
                <YAxis yAxisId="left" tick={{ fontSize: 10, fill: '#6b7280' }}
                       tickFormatter={v => `$${v}`} />
                <YAxis yAxisId="right" orientation="right" domain={[30, 70]}
                       tick={{ fontSize: 10, fill: '#fbbf24' }}
                       tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={{
                  backgroundColor: '#1a2235', border: '1px solid #243049',
                  borderRadius: 8, fontSize: 12, color: '#e5e7eb',
                }} />
                <Bar yAxisId="left" dataKey="pnl_usd" name="P&L ($)" radius={[3, 3, 0, 0]}
                     fill="#3b82f6" />
                <Line yAxisId="right" dataKey="win_rate" name="Win Rate %"
                      stroke="#fbbf24" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* A/B Test + Pair Economics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* A/B Test Results */}
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">A/B Direction Test</h3>
          {abSorted.length === 0 ? (
            <div className="text-sm text-gray-600 py-4 text-center">Collecting data...</div>
          ) : (
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="text-left py-2 px-1">Rule</th>
                  <th className="text-right py-2 px-1">Trades</th>
                  <th className="text-right py-2 px-1">Win Rate</th>
                  <th className="text-right py-2 px-1">P&L (bps)</th>
                </tr>
              </thead>
              <tbody>
                {abSorted.map(([rule, d], idx) => {
                  const wr = d.trades > 0 ? (d.wins / d.trades * 100).toFixed(1) : '0.0';
                  const isLeader = idx === 0;
                  return (
                    <tr key={rule} className={`border-b border-surface-3/50 ${isLeader ? 'bg-accent-green/5' : ''}`}>
                      <td className="py-2 px-1 text-gray-300">
                        <strong>{rule}</strong>{isLeader ? ' *' : ''}
                      </td>
                      <td className="py-2 px-1 text-right text-gray-400">{d.trades.toLocaleString()}</td>
                      <td className={`py-2 px-1 text-right ${parseFloat(wr) >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {wr}%
                      </td>
                      <td className={`py-2 px-1 text-right ${d.pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {d.pnl >= 0 ? '+' : ''}{d.pnl.toFixed(1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* Pair Economics */}
        <div className="lg:col-span-2 bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Pair Economics</h3>
          {pairEcon.length === 0 ? (
            <div className="text-sm text-gray-600 py-4 text-center">Collecting data...</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-gray-500 border-b border-surface-3">
                    <th className="text-left py-2 px-1">Pair</th>
                    <th className="text-left py-2 px-1">Rule</th>
                    <th className="text-right py-2 px-1">EV (bps)</th>
                    <th className="text-right py-2 px-1">Win Rate</th>
                    <th className="text-right py-2 px-1">Tokens</th>
                    <th className="text-right py-2 px-1">P&L ($)</th>
                    <th className="text-right py-2 px-1">Avg Hold</th>
                    <th className="text-center py-2 px-1">Active</th>
                  </tr>
                </thead>
                <tbody>
                  {pairEcon.map((p, i) => (
                    <tr key={i} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                      <td className="py-2 px-1 text-gray-300">{p.pair}</td>
                      <td className="py-2 px-1">
                        <span className="px-1.5 py-0.5 rounded bg-surface-2 text-gray-400">{p.direction_rule || '?'}</span>
                      </td>
                      <td className={`py-2 px-1 text-right ${p.current_ev >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {p.current_ev >= 0 ? '+' : ''}{p.current_ev.toFixed(1)}
                      </td>
                      <td className={`py-2 px-1 text-right ${p.win_rate >= 50 ? 'text-accent-green' : p.win_rate >= 47 ? 'text-accent-yellow' : 'text-accent-red'}`}>
                        {p.win_rate.toFixed(1)}%
                      </td>
                      <td className="py-2 px-1 text-right text-gray-400">{p.total_tokens.toLocaleString()}</td>
                      <td className={`py-2 px-1 text-right font-semibold ${(p.total_pnl_usd ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {(p.total_pnl_usd ?? 0) >= 0 ? '+' : ''}${(p.total_pnl_usd ?? 0).toFixed(2)}
                      </td>
                      <td className="py-2 px-1 text-right text-gray-500">
                        {p.avg_hold_sec != null ? `${p.avg_hold_sec.toFixed(0)}s` : '--'}
                      </td>
                      <td className="py-2 px-1 text-center">{p.tradeable ? 'Y' : 'N'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Live Token Feed */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">
          Live Token Feed <span className="text-gray-600">(last 30)</span>
        </h3>
        {feed.length === 0 ? (
          <div className="text-sm text-gray-600 py-4 text-center">Waiting for tokens...</div>
        ) : (
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-xs font-mono">
              <thead className="sticky top-0 bg-surface-1">
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="text-left py-2 px-1">Time</th>
                  <th className="text-left py-2 px-1">Pair</th>
                  <th className="text-left py-2 px-1">Dir</th>
                  <th className="text-left py-2 px-1">Rule</th>
                  <th className="text-right py-2 px-1">Size</th>
                  <th className="text-right py-2 px-1">P&L</th>
                  <th className="text-left py-2 px-1">Exit</th>
                  <th className="text-left py-2 px-1">Vol</th>
                </tr>
              </thead>
              <tbody>
                {feed.slice(0, 30).map((t, i) => {
                  const dirClass = t.direction === 'long' ? 'text-accent-green' : 'text-accent-red';
                  const dirLabel = t.direction === 'long' ? 'L' : 'S';
                  const pnlText = t.exit_pnl_bps != null
                    ? `${t.exit_pnl_bps >= 0 ? '+' : ''}${t.exit_pnl_bps.toFixed(1)}bp`
                    : 'open';
                  const pnlClass = t.exit_pnl_bps != null
                    ? (t.exit_pnl_bps >= 0 ? 'text-accent-green' : 'text-accent-red')
                    : 'text-gray-500';
                  return (
                    <tr key={i} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                      <td className="py-1.5 px-1 text-gray-500">{formatTime(t.timestamp)}</td>
                      <td className="py-1.5 px-1 text-gray-300">{t.pair}</td>
                      <td className={`py-1.5 px-1 ${dirClass}`}>{dirLabel}</td>
                      <td className="py-1.5 px-1">
                        <span className="px-1.5 py-0.5 rounded bg-surface-2 text-gray-400">{t.direction_rule || '?'}</span>
                      </td>
                      <td className="py-1.5 px-1 text-right text-gray-400">${(t.token_size_usd ?? 10).toFixed(0)}</td>
                      <td className={`py-1.5 px-1 text-right ${pnlClass}`}>{pnlText}</td>
                      <td className="py-1.5 px-1 text-gray-500">{shortenExitReason(t.exit_reason)}</td>
                      <td className="py-1.5 px-1">
                        <span className={`px-1.5 py-0.5 rounded text-xs ${VOL_COLORS[t.vol_regime] || 'bg-surface-2 text-gray-400'}`}>
                          {t.vol_regime || '?'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </PageShell>
  );
}
