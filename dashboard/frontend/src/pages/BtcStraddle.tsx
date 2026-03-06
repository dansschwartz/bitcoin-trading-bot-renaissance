import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import { api } from '../api';
import { Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Line, ComposedChart } from 'recharts';

interface FleetStatus {
  halted: boolean;
  fleet_daily_loss: number;
  fleet_daily_loss_limit: number;
  fleet_max_deployed: number;
  total_deployed: number;
  total_open: number;
  total_pnl: number;
  engines: Record<string, {
    open: number;
    deployed: number;
    pnl_usd: number;
    daily_loss: number;
    win_rate: number;
    total_closed: number;
  }>;
}

interface EngineStatus {
  active: boolean;
  observation_mode: boolean;
  asset: string;
  pair: string;
  size_usd: number;
  wallet_usd: number;
  interval_seconds: number;
  max_open: number;
  open_count: number;
  open_straddles: {
    straddle_id: number;
    asset: string;
    entry_price: number;
    vol_prediction: number | null;
    vol_ratio: number;
    effective_stop_bps: number;
    effective_activation_bps: number;
    effective_trail_bps: number;
    age_seconds: number;
    long_trail_active: boolean;
    short_trail_active: boolean;
    long_peak_bps: number;
    short_peak_bps: number;
  }[];
  total_opened: number;
  total_closed: number;
  total_winners: number;
  win_rate: number;
  total_pnl_usd: number;
  daily_loss_usd: number;
  max_daily_loss_usd: number;
  dead_zone_blocks: number;
  vol_scaling: string;
  config: Record<string, number | boolean | string>;
}

interface StraddleHistory {
  id: number;
  asset: string;
  opened_at: string;
  closed_at: string;
  entry_price: number;
  vol_prediction: number | null;
  vol_ratio: number;
  effective_stop_bps: number;
  long_exit_reason: string;
  short_exit_reason: string;
  long_pnl_bps: number;
  short_pnl_bps: number;
  net_pnl_bps: number;
  net_pnl_usd: number;
  size_usd: number;
  duration_seconds: number;
}

interface HourlyPnl {
  hour: string;
  asset: string;
  straddles: number;
  winners: number;
  pnl_usd: number;
  avg_pnl_bps: number;
  win_rate: number;
}

interface StraddleStats {
  total: number;
  winners: number;
  win_rate: number;
  total_pnl_usd: number;
  avg_net_bps: number;
  avg_duration: number;
  best_bps: number;
  worst_bps: number;
  avg_vol_ratio: number;
  exit_reasons: Record<string, number>;
  dead_zone_blocks: number;
  per_asset: { asset: string; total: number; winners: number; pnl_usd: number; avg_bps: number }[];
}

function formatTime(ts: string | null): string {
  if (!ts) return '--';
  try {
    const parts = ts.split(' ');
    return parts[1]?.substring(0, 8) || ts.substring(11, 19) || '--';
  } catch { return '--'; }
}

const ASSET_COLORS: Record<string, string> = {
  BTC: 'text-accent-yellow',
  ETH: 'text-accent-blue',
};

export default function BtcStraddle() {
  const [fleet, setFleet] = useState<FleetStatus | null>(null);
  const [engines, setEngines] = useState<EngineStatus[]>([]);
  const [history, setHistory] = useState<StraddleHistory[]>([]);
  const [hourly, setHourly] = useState<HourlyPnl[]>([]);
  const [stats, setStats] = useState<StraddleStats | null>(null);

  const refresh = () => {
    api.straddleFleet().then(d => setFleet(d as unknown as FleetStatus)).catch(() => {});
    api.straddleStatus().then(d => {
      const arr = Array.isArray(d) ? d : [d];
      setEngines(arr as unknown as EngineStatus[]);
    }).catch(() => {});
    api.straddleHistory(50).then(d => setHistory(d as unknown as StraddleHistory[])).catch(() => {});
    api.straddleHourly().then(d => setHourly([...(d as unknown as HourlyPnl[])].reverse())).catch(() => {});
    api.straddleStats().then(d => setStats(d as unknown as StraddleStats)).catch(() => {});
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  const exitReasons = stats?.exit_reasons || {};
  const totalReasons = Object.values(exitReasons).reduce((a, b) => a + b, 0);

  return (
    <PageShell title="Straddle Fleet" subtitle="Multi-asset direction-free paired LONG+SHORT — $500/leg, 5min hold, 35 concurrent">
      {/* Fleet Status Bar */}
      <div className={`border rounded-xl p-3 ${fleet?.halted ? 'bg-accent-red/10 border-accent-red/40' : 'bg-surface-1 border-surface-3'}`}>
        <div className="flex flex-wrap items-center gap-4 text-sm">
          {fleet?.halted && (
            <span className="inline-block px-2 py-0.5 rounded text-xs font-mono bg-accent-red/20 text-accent-red font-bold">
              HALTED
            </span>
          )}
          <span className="text-gray-400">Fleet P&L: <strong className={`${(fleet?.total_pnl ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {(fleet?.total_pnl ?? 0) >= 0 ? '+' : ''}${(fleet?.total_pnl ?? 0).toFixed(2)}
          </strong></span>
          <span className="text-gray-400">Deployed: <strong className="text-gray-200">
            ${(fleet?.total_deployed ?? 0).toLocaleString()} / ${(fleet?.fleet_max_deployed ?? 70000).toLocaleString()}
          </strong></span>
          <span className="text-gray-400">Open: <strong className="text-gray-200">{fleet?.total_open ?? 0}</strong></span>
          <span className="text-gray-400">Daily Loss: <strong className={`${(fleet?.fleet_daily_loss ?? 0) > 0 ? 'text-accent-red' : 'text-gray-200'}`}>
            ${(fleet?.fleet_daily_loss ?? 0).toFixed(2)} / ${(fleet?.fleet_daily_loss_limit ?? 1500).toFixed(0)}
          </strong></span>
        </div>
      </div>

      {/* Per-Asset Engine Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {engines.map((eng) => (
          <div key={eng.asset} className="bg-surface-1 border border-surface-3 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className={`text-lg font-bold ${ASSET_COLORS[eng.asset] || 'text-gray-200'}`}>
                  {eng.asset}
                </span>
                <span className={`inline-block px-2 py-0.5 rounded text-xs font-mono ${
                  eng.observation_mode ? 'bg-accent-yellow/20 text-accent-yellow' : 'bg-accent-green/20 text-accent-green'
                }`}>{eng.observation_mode ? 'OBS' : 'LIVE'}</span>
                {eng.vol_scaling === 'proportional' && (
                  <span className="inline-block px-2 py-0.5 rounded text-xs font-mono bg-accent-blue/20 text-accent-blue">
                    VOL-SCALED
                  </span>
                )}
              </div>
              <span className={`text-lg font-mono font-semibold ${(eng.total_pnl_usd ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                {(eng.total_pnl_usd ?? 0) >= 0 ? '+' : ''}${(eng.total_pnl_usd ?? 0).toFixed(2)}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs font-mono">
              <div><span className="text-gray-500">Size:</span> <span className="text-gray-300">${eng.size_usd}/leg</span></div>
              <div><span className="text-gray-500">Wallet:</span> <span className="text-gray-300">${(eng.wallet_usd ?? 7000).toLocaleString()}</span></div>
              <div><span className="text-gray-500">Open:</span> <span className="text-gray-300">{eng.open_count}/{eng.max_open}</span></div>
              <div><span className="text-gray-500">Closed:</span> <span className="text-gray-300">{eng.total_closed}</span></div>
              <div><span className="text-gray-500">Win Rate:</span> <span className={`${eng.win_rate >= 50 ? 'text-accent-green' : 'text-gray-300'}`}>{eng.win_rate?.toFixed(1)}%</span></div>
              <div><span className="text-gray-500">Daily Loss:</span> <span className={`${eng.daily_loss_usd > 0 ? 'text-accent-red' : 'text-gray-300'}`}>
                ${(eng.daily_loss_usd ?? 0).toFixed(2)}/{(eng.max_daily_loss_usd ?? 700).toFixed(0)}
              </span></div>
            </div>

            {/* Live open straddles for this engine */}
            {eng.open_straddles && eng.open_straddles.length > 0 && (
              <div className="mt-3 pt-3 border-t border-surface-3 space-y-2">
                {eng.open_straddles.map(s => (
                  <div key={s.straddle_id} className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
                    <div>
                      <span className="text-gray-500">#{s.straddle_id}</span>{' '}
                      <span className="text-gray-300">${s.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Age:</span>{' '}
                      <span className="text-gray-300">{s.age_seconds.toFixed(0)}s</span>
                      {' '}<span className="text-gray-600">vol_r={s.vol_ratio?.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-accent-green">L</span> {s.long_peak_bps.toFixed(1)}bp
                      {s.long_trail_active && <span className="ml-1 text-accent-yellow">TRAIL</span>}
                    </div>
                    <div>
                      <span className="text-accent-red">S</span> {s.short_peak_bps.toFixed(1)}bp
                      {s.short_trail_active && <span className="ml-1 text-accent-yellow">TRAIL</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        {[
          { label: 'Total', value: stats?.total ?? 0, fmt: (v: number) => v.toLocaleString() },
          { label: 'Win Rate', value: stats?.win_rate ?? 0, fmt: (v: number) => `${v.toFixed(1)}%`, color: (v: number) => v >= 50 ? 'text-accent-green' : 'text-accent-red' },
          { label: 'Net P&L', value: stats?.total_pnl_usd ?? 0, fmt: (v: number) => `${v >= 0 ? '+' : ''}$${v.toFixed(2)}`, color: (v: number) => v >= 0 ? 'text-accent-green' : 'text-accent-red' },
          { label: 'Avg Net', value: stats?.avg_net_bps ?? 0, fmt: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}bp`, color: (v: number) => v >= 0 ? 'text-accent-green' : 'text-accent-red' },
          { label: 'Avg Duration', value: stats?.avg_duration ?? 0, fmt: (v: number) => `${v.toFixed(0)}s` },
          { label: 'Avg Vol Ratio', value: stats?.avg_vol_ratio ?? 1, fmt: (v: number) => `${v.toFixed(2)}x` },
          { label: 'Best / Worst', value: 0, fmt: () => `${(stats?.best_bps ?? 0).toFixed(1)} / ${(stats?.worst_bps ?? 0).toFixed(1)}bp` },
        ].map((card, i) => (
          <div key={i} className="bg-surface-1 border border-surface-3 rounded-xl p-3 text-center">
            <div className="text-xs text-gray-500 mb-1">{card.label}</div>
            <div className={`text-lg font-mono font-semibold ${card.color ? card.color(card.value) : 'text-gray-200'}`}>
              {card.fmt(card.value)}
            </div>
          </div>
        ))}
      </div>

      {/* Per-Asset P&L Breakdown */}
      {stats?.per_asset && stats.per_asset.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {stats.per_asset.map(a => (
            <div key={a.asset} className="bg-surface-1 border border-surface-3 rounded-xl p-3">
              <div className="flex items-center justify-between">
                <span className={`font-bold ${ASSET_COLORS[a.asset] || 'text-gray-200'}`}>{a.asset}</span>
                <span className="text-xs text-gray-500">{a.total} trades</span>
              </div>
              <div className="flex items-center justify-between mt-1">
                <span className={`text-sm font-mono ${(a.pnl_usd ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                  {(a.pnl_usd ?? 0) >= 0 ? '+' : ''}${(a.pnl_usd ?? 0).toFixed(2)}
                </span>
                <span className="text-xs text-gray-400">
                  WR: {a.total > 0 ? ((a.winners / a.total) * 100).toFixed(1) : '0.0'}%
                  | Avg: {(a.avg_bps ?? 0) >= 0 ? '+' : ''}{(a.avg_bps ?? 0).toFixed(1)}bp
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Hourly P&L Chart */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Hourly P&L + Win Rate</h3>
        {hourly.length === 0 ? (
          <div className="text-sm text-gray-600 py-8 text-center">No hourly data yet -- waiting for closed straddles</div>
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

      {/* Exit Reasons + History */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Exit Reason Breakdown */}
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Exit Reasons</h3>
          {totalReasons === 0 ? (
            <div className="text-sm text-gray-600 py-4 text-center">No data yet</div>
          ) : (
            <div className="space-y-2">
              {Object.entries(exitReasons).sort((a, b) => b[1] - a[1]).map(([reason, count]) => {
                const pct = (count / totalReasons * 100).toFixed(1);
                const color = reason === 'trail_stop' ? 'bg-accent-green' :
                              reason === 'stop_loss' ? 'bg-accent-red' : 'bg-accent-yellow';
                return (
                  <div key={reason}>
                    <div className="flex justify-between text-xs mb-0.5">
                      <span className="text-gray-400 font-mono">{reason}</span>
                      <span className="text-gray-500">{count} ({pct}%)</span>
                    </div>
                    <div className="w-full h-1.5 bg-surface-3 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${color}`}
                           style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Recent History */}
        <div className="lg:col-span-2 bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">
            Recent Straddles <span className="text-gray-600">(last 50)</span>
          </h3>
          {history.length === 0 ? (
            <div className="text-sm text-gray-600 py-4 text-center">Waiting for closed straddles...</div>
          ) : (
            <div className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-xs font-mono">
                <thead className="sticky top-0 bg-surface-1">
                  <tr className="text-gray-500 border-b border-surface-3">
                    <th className="text-left py-2 px-1">Asset</th>
                    <th className="text-left py-2 px-1">Time</th>
                    <th className="text-right py-2 px-1">Entry</th>
                    <th className="text-right py-2 px-1">Vol-R</th>
                    <th className="text-left py-2 px-1">L Exit</th>
                    <th className="text-right py-2 px-1">L P&L</th>
                    <th className="text-left py-2 px-1">S Exit</th>
                    <th className="text-right py-2 px-1">S P&L</th>
                    <th className="text-right py-2 px-1">Net</th>
                    <th className="text-right py-2 px-1">Dur</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice(0, 50).map((s, i) => (
                    <tr key={i} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                      <td className={`py-1.5 px-1 font-semibold ${ASSET_COLORS[s.asset] || 'text-gray-300'}`}>
                        {s.asset || 'BTC'}
                      </td>
                      <td className="py-1.5 px-1 text-gray-500">{formatTime(s.closed_at)}</td>
                      <td className="py-1.5 px-1 text-right text-gray-300">
                        ${s.entry_price?.toLocaleString(undefined, { minimumFractionDigits: 0 })}
                      </td>
                      <td className="py-1.5 px-1 text-right text-gray-500">
                        {s.vol_ratio != null ? `${Number(s.vol_ratio).toFixed(2)}x` : '--'}
                      </td>
                      <td className="py-1.5 px-1">
                        <span className="px-1.5 py-0.5 rounded bg-surface-2 text-gray-400">
                          {s.long_exit_reason || '?'}
                        </span>
                      </td>
                      <td className={`py-1.5 px-1 text-right ${s.long_pnl_bps >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {s.long_pnl_bps >= 0 ? '+' : ''}{s.long_pnl_bps?.toFixed(1)}bp
                      </td>
                      <td className="py-1.5 px-1">
                        <span className="px-1.5 py-0.5 rounded bg-surface-2 text-gray-400">
                          {s.short_exit_reason || '?'}
                        </span>
                      </td>
                      <td className={`py-1.5 px-1 text-right ${s.short_pnl_bps >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {s.short_pnl_bps >= 0 ? '+' : ''}{s.short_pnl_bps?.toFixed(1)}bp
                      </td>
                      <td className={`py-1.5 px-1 text-right font-semibold ${s.net_pnl_bps >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {s.net_pnl_bps >= 0 ? '+' : ''}{s.net_pnl_bps?.toFixed(1)}bp
                      </td>
                      <td className="py-1.5 px-1 text-right text-gray-500">{s.duration_seconds?.toFixed(0)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}
