import { useState, useEffect } from 'react';
import PageShell from '../components/layout/PageShell';
import { api } from '../api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  LineChart, Line,
} from 'recharts';

// ─── Strategy colors (consistent across all sections) ───────────────────
const STRAT_COLORS: Record<string, string> = {
  cross_exchange: '#3b82f6',
  triangular: '#10b981',
  funding_rate: '#f59e0b',
};

const HEALTH_COLORS: Record<string, string> = {
  healthy: '#10b981',
  warning: '#f59e0b',
  critical: '#ef4444',
  dead: '#6b7280',
  insufficient_data: '#6b7280',
};

const HEALTH_ICONS: Record<string, string> = {
  healthy: '\u{1F7E2}',
  warning: '\u{1F7E1}',
  critical: '\u{1F534}',
  dead: '\u26AB',
  insufficient_data: '\u26AA',
};

// ─── Formatters ──────────────────────────────────────────────────────────
function fmtVelocity(v: number): string {
  if (v === 0) return '$0';
  const abs = Math.abs(v);
  if (abs >= 1000) return `$${(v / 1000).toFixed(1)}K`;
  if (abs >= 1) return `$${v.toFixed(2)}`;
  return `$${v.toFixed(4)}`;
}

function fmtTurns(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(1)}K`;
  return v.toFixed(0);
}

function fmtHoldTime(sec: number): string {
  if (sec < 60) return `${sec.toFixed(1)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.floor(sec % 60)}s`;
  return `${(sec / 3600).toFixed(1)}h`;
}

function timeAgo(ts: number | string): string {
  if (!ts) return 'never';
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  const sec = (Date.now() - d.getTime()) / 1000;
  if (sec < 60) return 'just now';
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

function stratLabel(s: string): string {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ─── Types ───────────────────────────────────────────────────────────────
type VelData = Record<string, unknown>;
type DecayData = Record<string, unknown>;
type AllocData = Record<string, unknown>;
type ExhaustData = Record<string, unknown>;

// ─── Chart tooltip styles ────────────────────────────────────────────────
const tooltipStyle = {
  contentStyle: { backgroundColor: '#1a2235', border: '1px solid #243049', borderRadius: 8 },
  labelStyle: { color: '#9ca3af' },
  itemStyle: { color: '#e5e7eb' },
};

// ═══════════════════════════════════════════════════════════════════════════
export default function Intelligence() {
  const [velocity, setVelocity] = useState<VelData | null>(null);
  const [decay, setDecay] = useState<DecayData | null>(null);
  const [allocation, setAllocation] = useState<AllocData | null>(null);
  const [exhaust, setExhaust] = useState<ExhaustData | null>(null);

  useEffect(() => {
    const load = () => {
      api.arbCapitalVelocity().then(d => setVelocity(d)).catch(() => {});
      api.arbEdgeDecay().then(d => setDecay(d)).catch(() => {});
      api.arbStrategyAllocation().then(d => setAllocation(d)).catch(() => {});
      api.arbExhaustSnapshots().then(d => setExhaust(d)).catch(() => {});
    };
    load();
    const id = setInterval(load, 60_000);
    return () => clearInterval(id);
  }, []);

  return (
    <PageShell
      title="Intelligence"
      subtitle="Capital velocity, edge decay, strategy allocation & data exhaust"
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <VelocitySection data={velocity} />
        <AllocationSection data={allocation} />
        <EdgeDecaySection data={decay} />
        <ExhaustSection data={exhaust} />
      </div>
    </PageShell>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: CAPITAL VELOCITY
// ═══════════════════════════════════════════════════════════════════════════
function VelocitySection({ data }: { data: VelData | null }) {
  const strategies = (data?.strategies ?? {}) as Record<string, Record<string, Record<string, number>>>;
  const hasData = Object.keys(strategies).length > 0;

  // Build chart data from 24h window
  const chartData = Object.entries(strategies).map(([name, windows]) => {
    const w24h = windows?.['24h'] ?? {};
    return {
      name: stratLabel(name),
      key: name,
      velocity: w24h.velocity ?? 0,
      trades: w24h.trades ?? 0,
      profit: w24h.total_profit ?? 0,
      capitalHours: w24h.total_capital_hours ?? 0,
    };
  }).sort((a, b) => b.velocity - a.velocity);

  // Build detailed table from all windows
  const tableData = Object.entries(strategies).map(([name, windows]) => {
    const w1h = windows?.['1h'] ?? {};
    const w24h = windows?.['24h'] ?? {};
    return {
      name,
      velocity1h: w1h.velocity ?? 0,
      velocity24h: w24h.velocity ?? 0,
      trades24h: w24h.trades ?? 0,
      profit24h: w24h.total_profit ?? 0,
    };
  }).sort((a, b) => b.velocity24h - a.velocity24h);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Capital Velocity</h3>

      {!hasData ? (
        <div className="text-xs text-gray-500 py-8 text-center">
          Insufficient data — velocity tracker is accumulating trade data...
        </div>
      ) : (
        <>
          {/* Bar chart */}
          <div className="h-40 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#243049" horizontal={false} />
                <XAxis
                  type="number"
                  scale="log"
                  domain={['auto', 'auto']}
                  tickFormatter={(v: number) => fmtVelocity(v)}
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                  axisLine={{ stroke: '#374151' }}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: '#e5e7eb', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={100}
                />
                <Tooltip
                  {...tooltipStyle}
                  formatter={(v: number) => [`${fmtVelocity(v)}/dol-hr`, 'Velocity']}
                />
                <Bar dataKey="velocity" radius={[0, 4, 4, 0]}>
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={STRAT_COLORS[d.key] ?? '#6b7280'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Stats table */}
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="text-left py-1.5 px-1">Strategy</th>
                <th className="text-right py-1.5 px-1">Vel (1h)</th>
                <th className="text-right py-1.5 px-1">Vel (24h)</th>
                <th className="text-right py-1.5 px-1">Trades</th>
                <th className="text-right py-1.5 px-1">24h Profit</th>
              </tr>
            </thead>
            <tbody>
              {tableData.map(row => (
                <tr key={row.name} className="border-b border-surface-3/50">
                  <td className="py-1.5 px-1 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: STRAT_COLORS[row.name] }} />
                    <span className="text-gray-300">{stratLabel(row.name)}</span>
                  </td>
                  <td className="py-1.5 px-1 text-right text-gray-400">{fmtVelocity(row.velocity1h)}</td>
                  <td className="py-1.5 px-1 text-right text-gray-300">{fmtVelocity(row.velocity24h)}</td>
                  <td className="py-1.5 px-1 text-right text-gray-400">{row.trades24h.toLocaleString()}</td>
                  <td className={`py-1.5 px-1 text-right ${row.profit24h > 0 ? 'text-accent-green' : 'text-gray-400'}`}>
                    ${row.profit24h.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Insight */}
          {tableData.length >= 2 && tableData[0].velocity24h > 0 && tableData[1].velocity24h > 0 && (
            <div className="mt-3 bg-accent-blue/10 border border-accent-blue/20 rounded-lg px-3 py-2 text-xs text-gray-300">
              {stratLabel(tableData[0].name)} is{' '}
              <span className="text-accent-blue font-medium">
                {(tableData[0].velocity24h / tableData[1].velocity24h).toFixed(0)}x
              </span>{' '}
              more capital-efficient than {stratLabel(tableData[1].name)}
            </div>
          )}
        </>
      )}

      <div className="mt-2 text-[10px] text-gray-600">
        Updated {data?.generated_at ? timeAgo(data.generated_at as number) : '—'}
        {' · '}{String(data?.total_tracked_trades ?? 0)} tracked trades
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: STRATEGY ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════
function AllocationSection({ data }: { data: AllocData | null }) {
  const targetAlloc = (data?.target_allocation ?? {}) as Record<string, number>;
  const currentAlloc = (data?.current_allocation ?? {}) as Record<string, number>;
  const scores = (data?.scores ?? {}) as Record<string, Record<string, number>>;
  const observationMode = Boolean(data?.observation_mode);
  const hasData = Object.keys(targetAlloc).length > 0;

  const pieData = Object.entries(targetAlloc).map(([name, pct]) => ({
    name: stratLabel(name),
    key: name,
    value: Math.round(pct * 1000) / 10, // Convert to percentage
  }));

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Strategy Allocation</h3>
        <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
          observationMode
            ? 'bg-amber-400/20 text-amber-400'
            : 'bg-green-400/20 text-green-400'
        }`}>
          {observationMode ? 'OBSERVATION' : 'ACTIVE'}
        </span>
      </div>

      {!hasData ? (
        <div className="text-xs text-gray-500 py-8 text-center">
          Allocator has not run yet — waiting for first weekly cycle
        </div>
      ) : (
        <>
          {/* Donut chart */}
          <div className="h-48 mb-3">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={75}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {pieData.map((d, i) => (
                    <Cell key={i} fill={STRAT_COLORS[d.key] ?? '#6b7280'} stroke="none" />
                  ))}
                </Pie>
                <Legend
                  iconType="circle"
                  iconSize={8}
                  formatter={(value: string) => <span className="text-xs text-gray-400">{value}</span>}
                />
                <Tooltip
                  {...tooltipStyle}
                  formatter={(v: number) => [`${v.toFixed(1)}%`, 'Target']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Allocation bars */}
          <div className="space-y-2.5">
            {Object.entries(targetAlloc).sort((a, b) => b[1] - a[1]).map(([name, pct]) => {
              const pctDisplay = (pct * 100).toFixed(1);
              const curPct = ((currentAlloc[name] ?? pct) * 100).toFixed(1);
              const score = scores[name];
              return (
                <div key={name}>
                  <div className="flex justify-between text-xs mb-0.5">
                    <span className="text-gray-400 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: STRAT_COLORS[name] }} />
                      {stratLabel(name)}
                    </span>
                    <span className="font-mono text-gray-300">
                      {pctDisplay}%
                      {score && <span className="text-gray-600 ml-2">score: {(score.composite ?? 0).toFixed(2)}</span>}
                    </span>
                  </div>
                  <div className="w-full bg-surface-3 rounded-full h-2 relative">
                    {/* Current allocation (faded) */}
                    <div
                      className="absolute top-0 left-0 h-2 rounded-full opacity-30"
                      style={{
                        width: `${Math.min(100, Number(curPct))}%`,
                        backgroundColor: STRAT_COLORS[name] ?? '#6b7280',
                      }}
                    />
                    {/* Target allocation */}
                    <div
                      className="absolute top-0 left-0 h-2 rounded-full"
                      style={{
                        width: `${Math.min(100, Number(pctDisplay))}%`,
                        backgroundColor: STRAT_COLORS[name] ?? '#6b7280',
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Score breakdown */}
          {Object.keys(scores).length > 0 && (
            <div className="mt-3 pt-3 border-t border-surface-3">
              <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5">Scoring Weights: 50% velocity · 30% health · 20% sharpe</div>
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(scores).sort((a, b) => (b[1].composite ?? 0) - (a[1].composite ?? 0)).map(([name, s]) => (
                  <div key={name} className="bg-surface-2 rounded-lg p-2">
                    <div className="text-[10px] text-gray-500">{stratLabel(name)}</div>
                    <div className="text-xs font-mono text-gray-300 mt-0.5">
                      Sharpe: {(s.sharpe ?? 0).toFixed(1)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      <div className="mt-2 text-[10px] text-gray-600">
        Updated {data?.generated_at ? timeAgo(data.generated_at as number) : '—'}
        {observationMode && ' · Observation mode (first 2 weeks)'}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: EDGE DECAY
// ═══════════════════════════════════════════════════════════════════════════
function EdgeDecaySection({ data }: { data: DecayData | null }) {
  const strategies = (data?.strategies ?? {}) as Record<string, Record<string, unknown>>;
  const hasData = Object.keys(strategies).length > 0;

  // Build line chart data from daily_data
  const chartLines: Array<{ date: string; [key: string]: number | string }> = [];
  const dateSet = new Set<string>();

  // Collect all dates
  for (const [, strat] of Object.entries(strategies)) {
    const daily = (strat.daily_data ?? []) as Array<{ date: string; avg_profit: number }>;
    for (const d of daily) dateSet.add(d.date);
  }

  const dates = Array.from(dateSet).sort();
  for (const date of dates) {
    const point: { date: string; [key: string]: number | string } = { date: date.slice(5) }; // "03-15"
    for (const [name, strat] of Object.entries(strategies)) {
      const daily = (strat.daily_data ?? []) as Array<{ date: string; avg_profit: number }>;
      const found = daily.find(d => d.date === date);
      if (found) point[name] = found.avg_profit;
    }
    chartLines.push(point);
  }

  // Find strategies with warning/critical health
  const alerts = Object.entries(strategies)
    .filter(([, s]) => s.health === 'warning' || s.health === 'critical')
    .map(([name, s]) => {
      const slope = Number(s.slope ?? 0);
      const current = Number(s.latest_avg_profit ?? s.avg_daily_profit ?? 0);
      const weeksToZero = slope < 0 && current > 0 ? current / Math.abs(slope) : null;
      const zeroDate = weeksToZero != null
        ? new Date(Date.now() + weeksToZero * 7 * 86400 * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
        : null;
      return { name, health: s.health as string, slope, current, zeroDate, weeksToZero };
    });

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Edge Decay</h3>

      {!hasData ? (
        <div className="text-xs text-gray-500 py-8 text-center">
          Insufficient data — need at least 3 days of trading history
        </div>
      ) : (
        <>
          {/* Line chart: daily avg profit per strategy */}
          {chartLines.length > 0 && (
            <div className="h-48 mb-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartLines} margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                    axisLine={{ stroke: '#374151' }}
                  />
                  <YAxis
                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                    axisLine={{ stroke: '#374151' }}
                    tickFormatter={(v: number) => `$${v.toFixed(2)}`}
                    label={{ value: 'Avg $/trade', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 10 }}
                  />
                  <Tooltip
                    {...tooltipStyle}
                    formatter={(v: number, name: string) => [`$${v.toFixed(4)}`, stratLabel(name)]}
                  />
                  {Object.keys(strategies).filter(s => strategies[s].daily_data).map(name => (
                    <Line
                      key={name}
                      type="monotone"
                      dataKey={name}
                      stroke={STRAT_COLORS[name] ?? '#6b7280'}
                      strokeWidth={2}
                      dot={{ r: 3, fill: STRAT_COLORS[name] ?? '#6b7280' }}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Health badges */}
          <div className="space-y-2 mb-3">
            {Object.entries(strategies).map(([name, strat]) => {
              const health = String(strat.health ?? 'insufficient_data');
              const slope = Number(strat.slope ?? 0);
              const daysAnalyzed = Number(strat.days_analyzed ?? 0);
              return (
                <div key={name} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: STRAT_COLORS[name] }} />
                    <span className="text-xs text-gray-300">{stratLabel(name)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-gray-400">
                      {daysAnalyzed > 0 ? `${slope >= 0 ? '+' : ''}${slope.toFixed(4)}/day` : 'n/a'}
                    </span>
                    <span
                      className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                      style={{
                        backgroundColor: `${HEALTH_COLORS[health]}20`,
                        color: HEALTH_COLORS[health],
                      }}
                    >
                      {HEALTH_ICONS[health]} {health.toUpperCase()}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Alerts */}
          {alerts.map(a => (
            <div
              key={a.name}
              className={`rounded-lg px-3 py-2 text-xs mb-2 ${
                a.health === 'critical'
                  ? 'bg-red-500/10 border border-red-500/20 text-red-300'
                  : 'bg-amber-500/10 border border-amber-500/20 text-amber-300'
              }`}
            >
              <div className="font-medium">
                {a.health === 'critical' ? '\u{1F6A8}' : '\u26A0\uFE0F'}{' '}
                {stratLabel(a.name)} edge declining at {Math.abs(a.slope).toFixed(4)}/day
              </div>
              <div className="text-gray-400 mt-0.5">
                Current avg: ${a.current.toFixed(4)}/trade
                {a.zeroDate && (
                  <span> · At this rate, edge hits zero around <span className="text-gray-300">{a.zeroDate}</span></span>
                )}
              </div>
            </div>
          ))}
        </>
      )}

      <div className="mt-2 text-[10px] text-gray-600">
        {data?.window_days ? `${data.window_days}-day window` : ''}
        {' · '}Updated {data?.generated_at ? timeAgo(data.generated_at as number) : '—'}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: DATA EXHAUST
// ═══════════════════════════════════════════════════════════════════════════
function ExhaustSection({ data }: { data: ExhaustData | null }) {
  const snapshots = (data?.snapshots ?? []) as Array<Record<string, unknown>>;
  const count = Number(data?.count ?? 0);

  // Compute latency and market impact from snapshots
  // Group by signal_id to find detection → execution pairs
  const bySignal: Record<string, Array<Record<string, unknown>>> = {};
  for (const s of snapshots) {
    const id = String(s.signal_id ?? '');
    if (!id) continue;
    if (!bySignal[id]) bySignal[id] = [];
    bySignal[id].push(s);
  }

  // Compute spread decay: compare detection vs execution book snapshots
  let spreadDecayCount = 0;
  let totalSpreadDecay = 0;
  let totalLatency = 0;
  let latencyCount = 0;
  const recentSignals: Array<{
    signal_id: string; symbol: string; phase_count: number;
    detection_spread_bps: number | null; execution_spread_bps: number | null;
    post_spread_bps: number | null; latency_ms: number | null;
  }> = [];

  for (const [sigId, snaps] of Object.entries(bySignal)) {
    const det = snaps.filter(s => s.phase === 'detection');
    const exec = snaps.filter(s => s.phase === 'execution');
    const post = snaps.filter(s => s.phase === 'post_execution');

    const symbol = String(snaps[0]?.symbol ?? '');

    // Compute spread from book snapshot (best ask - best bid in bps)
    const computeSpread = (group: Array<Record<string, unknown>>): number | null => {
      if (group.length < 2) return null;
      // Find the two exchange sides
      const bids: number[] = [];
      const asks: number[] = [];
      for (const snap of group) {
        const b1 = Number(snap.bid_1 ?? 0);
        const a1 = Number(snap.ask_1 ?? 0);
        if (b1 > 0) bids.push(b1);
        if (a1 > 0) asks.push(a1);
      }
      if (bids.length < 1 || asks.length < 1) return null;
      const bestBid = Math.max(...bids);
      const bestAsk = Math.min(...asks);
      const mid = (bestBid + bestAsk) / 2;
      if (mid <= 0) return null;
      return ((bestBid - bestAsk) / mid) * 10000; // Can be negative (bid > ask = arb opportunity)
    };

    const detSpread = computeSpread(det);
    const execSpread = computeSpread(exec);
    const postSpread = computeSpread(post);

    // Latency: time between detection and execution
    if (det.length > 0 && exec.length > 0) {
      const detTime = new Date(String(det[0].timestamp)).getTime();
      const execTime = new Date(String(exec[0].timestamp)).getTime();
      const lat = execTime - detTime;
      if (lat > 0 && lat < 60000) {
        totalLatency += lat;
        latencyCount++;
      }
    }

    // Spread decay
    if (detSpread != null && execSpread != null) {
      totalSpreadDecay += Math.abs(detSpread) - Math.abs(execSpread);
      spreadDecayCount++;
    }

    recentSignals.push({
      signal_id: sigId.slice(-12),
      symbol,
      phase_count: snaps.length,
      detection_spread_bps: detSpread,
      execution_spread_bps: execSpread,
      post_spread_bps: postSpread,
      latency_ms: (det.length > 0 && exec.length > 0)
        ? new Date(String(exec[0].timestamp)).getTime() - new Date(String(det[0].timestamp)).getTime()
        : null,
    });
  }

  const avgLatency = latencyCount > 0 ? totalLatency / latencyCount : 0;
  const avgSpreadDecay = spreadDecayCount > 0 ? totalSpreadDecay / spreadDecayCount : 0;
  const signalCount = Object.keys(bySignal).length;

  // Market impact: compare execution vs post-execution spreads
  let impactTotal = 0;
  let impactCount = 0;
  for (const sig of recentSignals) {
    if (sig.execution_spread_bps != null && sig.post_spread_bps != null) {
      impactTotal += Math.abs(sig.execution_spread_bps) - Math.abs(sig.post_spread_bps);
      impactCount++;
    }
  }
  const avgImpact = impactCount > 0 ? impactTotal / impactCount : 0;

  const latencyColor = avgLatency < 50 ? 'text-accent-green' : avgLatency < 200 ? 'text-accent-yellow' : 'text-accent-red';
  const decayColor = Math.abs(avgSpreadDecay) < 1 ? 'text-accent-green' : Math.abs(avgSpreadDecay) < 3 ? 'text-accent-yellow' : 'text-accent-red';
  const impactColor = Math.abs(avgImpact) < 1 ? 'text-accent-green' : Math.abs(avgImpact) < 3 ? 'text-accent-yellow' : 'text-accent-red';

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Data Exhaust</h3>

      {count === 0 ? (
        <div className="text-xs text-gray-500 py-8 text-center">
          No exhaust snapshots yet — data will populate as cross-exchange signals fire
        </div>
      ) : (
        <>
          {/* Latency metrics */}
          <div className="grid grid-cols-3 gap-3 mb-4">
            <div className="bg-surface-2 rounded-lg p-3 text-center">
              <div className={`text-xl font-mono font-semibold ${latencyColor}`}>
                {avgLatency > 0 ? `${avgLatency.toFixed(0)}` : '—'}
                <span className="text-xs ml-0.5">ms</span>
              </div>
              <div className="text-[10px] text-gray-500 mt-0.5">Avg Latency</div>
            </div>
            <div className="bg-surface-2 rounded-lg p-3 text-center">
              <div className={`text-xl font-mono font-semibold ${decayColor}`}>
                {spreadDecayCount > 0 ? `${avgSpreadDecay.toFixed(1)}` : '—'}
                <span className="text-xs ml-0.5">bps</span>
              </div>
              <div className="text-[10px] text-gray-500 mt-0.5">Spread Decay</div>
            </div>
            <div className="bg-surface-2 rounded-lg p-3 text-center">
              <div className={`text-xl font-mono font-semibold ${impactColor}`}>
                {impactCount > 0 ? `${avgImpact >= 0 ? '+' : ''}${avgImpact.toFixed(1)}` : '—'}
                <span className="text-xs ml-0.5">bps</span>
              </div>
              <div className="text-[10px] text-gray-500 mt-0.5">Market Impact</div>
            </div>
          </div>

          {/* Recent signal snapshots */}
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5">Recent Signal Snapshots</div>
          <div className="overflow-x-auto max-h-48 overflow-y-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-gray-500 border-b border-surface-3 sticky top-0 bg-surface-1">
                  <th className="text-left py-1 px-1">Signal</th>
                  <th className="text-left py-1 px-1">Symbol</th>
                  <th className="text-right py-1 px-1">Phases</th>
                  <th className="text-right py-1 px-1">Det Spread</th>
                  <th className="text-right py-1 px-1">Exec Spread</th>
                  <th className="text-right py-1 px-1">Latency</th>
                </tr>
              </thead>
              <tbody>
                {recentSignals.slice(0, 20).map((sig, i) => (
                  <tr key={i} className="border-b border-surface-3/50">
                    <td className="py-1 px-1 text-gray-500">...{sig.signal_id}</td>
                    <td className="py-1 px-1 text-gray-300">{sig.symbol}</td>
                    <td className="py-1 px-1 text-right text-gray-400">{sig.phase_count}</td>
                    <td className="py-1 px-1 text-right text-gray-400">
                      {sig.detection_spread_bps != null ? `${sig.detection_spread_bps.toFixed(1)}` : '—'}
                    </td>
                    <td className="py-1 px-1 text-right text-gray-400">
                      {sig.execution_spread_bps != null ? `${sig.execution_spread_bps.toFixed(1)}` : '—'}
                    </td>
                    <td className="py-1 px-1 text-right text-gray-400">
                      {sig.latency_ms != null ? `${sig.latency_ms}ms` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <div className="mt-2 text-[10px] text-gray-600">
        Based on {count.toLocaleString()} snapshots across {signalCount} signals
      </div>
    </div>
  );
}
