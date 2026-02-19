import { useEffect, useState, useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Line, ComposedChart } from 'recharts';
import { api } from '../../api';
import type { EquityPoint, TimeRange } from '../../types';
import TimeRangeSelector from '../shared/TimeRangeSelector';
import { formatCurrency, formatTimestamp } from '../../utils/formatters';

interface BenchmarkPoint {
  timestamp: string;
  benchmark_equity: number;
}

export default function EquityCurve() {
  const [range, setRange] = useState<TimeRange>('1D');
  const [data, setData] = useState<EquityPoint[]>([]);
  const [benchmark, setBenchmark] = useState<BenchmarkPoint[]>([]);
  const [showBenchmark, setShowBenchmark] = useState(true);

  useEffect(() => {
    api.equity(range).then(setData).catch(() => {});
    api.benchmark(range, 'BTC-USD').then(setBenchmark).catch(() => setBenchmark([]));
  }, [range]);

  // Merge equity data with benchmark data by timestamp proximity
  const merged = useMemo(() => {
    if (!data.length) return [];
    const hasEquity = data[0]?.equity != null;
    const dataKey = hasEquity ? 'equity' : 'cumulative_pnl';

    // Create merged dataset
    const result = data.map(d => ({
      timestamp: d.timestamp,
      equity: hasEquity ? (d.equity ?? 0) : d.cumulative_pnl,
      benchmark_equity: undefined as number | undefined,
    }));

    // If we have benchmark data, merge it in
    if (benchmark.length > 0 && showBenchmark) {
      // Simple approach: for each equity point, find nearest benchmark point
      let bIdx = 0;
      for (const r of result) {
        const ts = new Date(r.timestamp).getTime();
        while (bIdx < benchmark.length - 1 && new Date(benchmark[bIdx + 1].timestamp).getTime() <= ts) {
          bIdx++;
        }
        if (bIdx < benchmark.length) {
          r.benchmark_equity = benchmark[bIdx].benchmark_equity;
        }
      }
    }

    return result;
  }, [data, benchmark, showBenchmark]);

  const hasEquity = data.length > 0 && data[0]?.equity != null;
  const initialCapital = hasEquity ? 10000 : 0;
  const lastEquity = merged.length > 0 ? (merged[merged.length - 1]?.equity ?? 0) : 0;
  const lastBenchmark = merged.length > 0 ? (merged[merged.length - 1]?.benchmark_equity ?? initialCapital) : initialCapital;
  const isPositive = lastEquity >= initialCapital;
  const beatingBenchmark = lastEquity > lastBenchmark;

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-medium text-gray-300">Equity Curve</h3>
          {benchmark.length > 0 && (
            <button
              onClick={() => setShowBenchmark(b => !b)}
              className={`text-[10px] px-1.5 py-0.5 rounded transition-colors ${
                showBenchmark ? 'bg-blue-500/20 text-blue-400' : 'text-gray-600 hover:text-gray-400'
              }`}
            >
              BTC B&H
            </button>
          )}
        </div>
        <div className="flex items-center gap-3">
          {hasEquity && merged.length > 0 && (
            <div className="flex items-center gap-2">
              <span className={`text-xs font-mono ${isPositive ? 'text-accent-green' : 'text-accent-red'}`}>
                {formatCurrency(lastEquity)}
              </span>
              {showBenchmark && benchmark.length > 0 && (
                <span className={`text-[10px] font-mono ${beatingBenchmark ? 'text-accent-green' : 'text-accent-red'}`}>
                  vs BTC {formatCurrency(lastBenchmark)}
                </span>
              )}
            </div>
          )}
          <TimeRangeSelector value={range} onChange={setRange} />
        </div>
      </div>
      <div className="h-56">
        {merged.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={merged} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
              <defs>
                <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={isPositive ? '#00d395' : '#ff4757'} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={isPositive ? '#00d395' : '#ff4757'} stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatTimestamp}
                tick={{ fontSize: 10, fill: '#6b7280' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tickFormatter={(v: number) => formatCurrency(v)}
                tick={{ fontSize: 10, fill: '#6b7280' }}
                axisLine={false}
                tickLine={false}
                width={60}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #243049', borderRadius: 8, fontSize: 12, color: '#e5e7eb' }}
                labelFormatter={formatTimestamp}
                formatter={(value: number, name: string) => [
                  formatCurrency(value),
                  name === 'equity' ? 'Strategy' : 'BTC Buy & Hold',
                ]}
              />
              {hasEquity && (
                <ReferenceLine y={initialCapital} stroke="#374151" strokeDasharray="3 3" />
              )}
              <Area
                type="monotone"
                dataKey="equity"
                stroke={isPositive ? '#00d395' : '#ff4757'}
                fill="url(#eqGrad)"
                strokeWidth={2}
              />
              {showBenchmark && benchmark.length > 0 && (
                <Line
                  type="monotone"
                  dataKey="benchmark_equity"
                  stroke="#3b82f6"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                  connectNulls
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-gray-600 text-sm">
            No equity data for this range
          </div>
        )}
      </div>
    </div>
  );
}
