import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import { api } from '../api';
import { Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Line, ComposedChart } from 'recharts';

interface ScannerStatus {
  enabled: boolean;
  running: boolean;
  observation_mode: boolean;
  scan_interval_seconds: number;
  total_scans: number;
  triggers_fired: number;
  positions_exited?: number;
  trigger_counts?: Record<string, number>;
  symbols_tracked?: number;
  active_cooldowns?: number;
}

interface ExitPerf {
  exit_reason: string;
  count: number;
  winners: number;
  win_rate: number;
  avg_pnl_bps: number;
  total_pnl_usd: number;
  avg_hold_sec: number;
}

interface SubBarEvent {
  reason: string;
  count: number;
  avg_pnl_bps: number;
}

interface RecentExit {
  timestamp: string;
  pair: string;
  direction: string;
  exit_reason: string;
  exit_pnl_bps: number | null;
  exit_pnl_usd: number | null;
  hold_time_seconds: number | null;
  token_size_usd: number | null;
}

interface HoldBucket {
  bucket: string;
  count: number;
  avg_pnl_bps: number;
  winners?: number;
}

function shortenReason(reason: string | null): string {
  if (!reason) return '--';
  return reason
    .replace('EDGE_CONSUMED_EARLY', 'EDGE_CONSUMED')
    .replace('STOP_LOSS_EARLY', 'STOP_LOSS');
}

function formatTime(ts: string | null): string {
  if (!ts) return '--';
  try {
    const parts = ts.split(' ');
    return parts[1]?.substring(0, 8) || ts.substring(11, 19) || '--';
  } catch { return '--'; }
}

export default function ExitEngine() {
  const [scanner, setScanner] = useState<ScannerStatus | null>(null);
  const [exitPerf, setExitPerf] = useState<ExitPerf[]>([]);
  const [subBarEvents, setSubBarEvents] = useState<SubBarEvent[]>([]);
  const [recentExits, setRecentExits] = useState<RecentExit[]>([]);
  const [holdDist, setHoldDist] = useState<HoldBucket[]>([]);

  const refresh = () => {
    api.exitSummary().then(d => {
      const data = d as Record<string, unknown>;
      setScanner((data.scanner as ScannerStatus) || null);
      setExitPerf((data.exit_performance as ExitPerf[]) || []);
      setSubBarEvents((data.sub_bar_events as SubBarEvent[]) || []);
    }).catch(() => {});
    api.exitRecentExits().then(d => setRecentExits(d as unknown as RecentExit[])).catch(() => {});
    api.exitHoldTimeDist().then(d => setHoldDist(d as unknown as HoldBucket[])).catch(() => {});
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10000);
    return () => clearInterval(id);
  }, []);

  const triggerRate = scanner && scanner.total_scans > 0
    ? ((scanner.triggers_fired || 0) / scanner.total_scans * 100).toFixed(1)
    : '0.0';

  return (
    <PageShell title="Exit Engine" subtitle="Sub-bar scanner performance, exit analysis, hold time distribution">
      {/* Scanner Status */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-3">
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <span>Sub-Bar: <span className={`inline-block px-2 py-0.5 rounded text-xs font-mono ${
            scanner?.running ? 'bg-accent-green/20 text-accent-green' : 'bg-accent-red/20 text-accent-red'
          }`}>{scanner?.running ? 'Active' : 'Stopped'}</span></span>
          <span className="text-gray-400">Cycle: <strong className="text-gray-200">{scanner?.scan_interval_seconds ?? 10}s</strong></span>
          <span className="text-gray-400">Scans: <strong className="text-gray-200">{(scanner?.total_scans ?? 0).toLocaleString()}</strong></span>
          <span className="text-gray-400">Triggers: <strong className="text-gray-200">{(scanner?.triggers_fired ?? 0).toLocaleString()}</strong></span>
          <span className="text-gray-400">Exited: <strong className="text-gray-200">{(scanner?.positions_exited ?? 0).toLocaleString()}</strong></span>
          <span className="text-gray-400">Trigger Rate: <strong className="text-accent-yellow">{triggerRate}%</strong></span>
          {scanner?.observation_mode && (
            <span className="px-2 py-0.5 rounded text-xs font-mono bg-accent-yellow/20 text-accent-yellow">OBSERVATION</span>
          )}
        </div>
      </div>

      {/* Trigger Type Breakdown (if available) */}
      {scanner?.trigger_counts && Object.keys(scanner.trigger_counts).length > 0 && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-3">
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <span className="text-gray-500 font-medium">Triggers:</span>
            {Object.entries(scanner.trigger_counts).map(([type, count]) => (
              <span key={type} className="px-2 py-1 rounded bg-surface-2 text-gray-300 font-mono">
                {shortenReason(type)}: <strong>{count}</strong>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Exit Performance + Hold Time side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* Exit Performance by Reason */}
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Exit Performance by Reason</h3>
          {exitPerf.length === 0 && subBarEvents.length === 0 ? (
            <div className="text-sm text-gray-600 py-4 text-center">No exit data yet</div>
          ) : (
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="text-left py-2 px-1">Reason</th>
                  <th className="text-right py-2 px-1">Count</th>
                  <th className="text-right py-2 px-1">Win Rate</th>
                  <th className="text-right py-2 px-1">Avg P&L</th>
                  <th className="text-right py-2 px-1">Total P&L</th>
                  <th className="text-right py-2 px-1">Avg Hold</th>
                </tr>
              </thead>
              <tbody>
                {exitPerf.map((r, i) => (
                  <tr key={i} className="border-b border-surface-3/50">
                    <td className="py-2 px-1 text-gray-300">{shortenReason(r.exit_reason)}</td>
                    <td className="py-2 px-1 text-right text-gray-400">{r.count}</td>
                    <td className={`py-2 px-1 text-right ${r.win_rate >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {r.win_rate.toFixed(1)}%
                    </td>
                    <td className={`py-2 px-1 text-right ${(r.avg_pnl_bps ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {(r.avg_pnl_bps ?? 0) >= 0 ? '+' : ''}{(r.avg_pnl_bps ?? 0).toFixed(1)}bp
                    </td>
                    <td className={`py-2 px-1 text-right font-semibold ${(r.total_pnl_usd ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      ${(r.total_pnl_usd ?? 0).toFixed(2)}
                    </td>
                    <td className="py-2 px-1 text-right text-gray-500">
                      {r.avg_hold_sec != null ? `${r.avg_hold_sec.toFixed(0)}s` : '--'}
                    </td>
                  </tr>
                ))}
                {/* Sub-bar events as separate section */}
                {subBarEvents.length > 0 && (
                  <>
                    <tr><td colSpan={6} className="py-1 text-gray-600 text-xs">Sub-Bar Events</td></tr>
                    {subBarEvents.map((e, i) => (
                      <tr key={`sb-${i}`} className="border-b border-surface-3/50">
                        <td className="py-2 px-1 text-gray-400">{shortenReason(e.reason)}</td>
                        <td className="py-2 px-1 text-right text-gray-400">{e.count}</td>
                        <td className="py-2 px-1 text-right text-gray-500">--</td>
                        <td className={`py-2 px-1 text-right ${(e.avg_pnl_bps ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                          {(e.avg_pnl_bps ?? 0) >= 0 ? '+' : ''}{(e.avg_pnl_bps ?? 0).toFixed(1)}bp
                        </td>
                        <td className="py-2 px-1 text-right text-gray-500">--</td>
                        <td className="py-2 px-1 text-right text-gray-500">--</td>
                      </tr>
                    ))}
                  </>
                )}
              </tbody>
            </table>
          )}
        </div>

        {/* Hold Time Distribution Chart */}
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Hold Time Distribution</h3>
          {holdDist.length === 0 ? (
            <div className="text-sm text-gray-600 py-8 text-center">No hold time data yet</div>
          ) : (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={holdDist} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <XAxis dataKey="bucket" tick={{ fontSize: 10, fill: '#6b7280' }} />
                  <YAxis yAxisId="left" tick={{ fontSize: 10, fill: '#6b7280' }} />
                  <YAxis yAxisId="right" orientation="right"
                         tick={{ fontSize: 10, fill: '#fbbf24' }}
                         tickFormatter={v => `${v}bp`} />
                  <Tooltip contentStyle={{
                    backgroundColor: '#1a2235', border: '1px solid #243049',
                    borderRadius: 8, fontSize: 12, color: '#e5e7eb',
                  }} />
                  <Bar yAxisId="left" dataKey="count" name="Exits" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                  <Line yAxisId="right" dataKey="avg_pnl_bps" name="Avg P&L (bps)"
                        stroke="#fbbf24" strokeWidth={2} dot={{ r: 3 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* Recent Exits Feed */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">
          Recent Exits <span className="text-gray-600">(last 50)</span>
        </h3>
        {recentExits.length === 0 ? (
          <div className="text-sm text-gray-600 py-4 text-center">No exits recorded yet</div>
        ) : (
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-xs font-mono">
              <thead className="sticky top-0 bg-surface-1">
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="text-left py-2 px-1">Time</th>
                  <th className="text-left py-2 px-1">Pair</th>
                  <th className="text-left py-2 px-1">Dir</th>
                  <th className="text-left py-2 px-1">Reason</th>
                  <th className="text-right py-2 px-1">P&L (bps)</th>
                  <th className="text-right py-2 px-1">P&L ($)</th>
                  <th className="text-right py-2 px-1">Hold</th>
                  <th className="text-right py-2 px-1">Size</th>
                </tr>
              </thead>
              <tbody>
                {recentExits.map((e, i) => {
                  const dirClass = e.direction === 'long' || e.direction === 'BUY' ? 'text-accent-green' : 'text-accent-red';
                  const dirLabel = e.direction === 'long' || e.direction === 'BUY' ? 'L' : 'S';
                  return (
                    <tr key={i} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                      <td className="py-1.5 px-1 text-gray-500">{formatTime(e.timestamp)}</td>
                      <td className="py-1.5 px-1 text-gray-300">{e.pair}</td>
                      <td className={`py-1.5 px-1 ${dirClass}`}>{dirLabel}</td>
                      <td className="py-1.5 px-1 text-gray-400">{shortenReason(e.exit_reason)}</td>
                      <td className={`py-1.5 px-1 text-right ${(e.exit_pnl_bps ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {e.exit_pnl_bps != null ? `${e.exit_pnl_bps >= 0 ? '+' : ''}${e.exit_pnl_bps.toFixed(1)}` : '--'}
                      </td>
                      <td className={`py-1.5 px-1 text-right ${(e.exit_pnl_usd ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {e.exit_pnl_usd != null ? `$${e.exit_pnl_usd.toFixed(3)}` : '--'}
                      </td>
                      <td className="py-1.5 px-1 text-right text-gray-500">
                        {e.hold_time_seconds != null ? `${e.hold_time_seconds.toFixed(0)}s` : '--'}
                      </td>
                      <td className="py-1.5 px-1 text-right text-gray-500">
                        {e.token_size_usd != null ? `$${e.token_size_usd.toFixed(0)}` : '--'}
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
