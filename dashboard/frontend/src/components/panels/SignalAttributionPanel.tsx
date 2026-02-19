import { useEffect, useState } from 'react';
import { api } from '../../api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface SignalStat {
  signal: string;
  trade_count: number;
  avg_contribution: number;
  total_contribution: number;
  abs_total: number;
  buy_count: number;
  sell_count: number;
}

interface AssetBreakdown {
  product_id: string;
  top_signal: string;
  top_signal_value: number;
  signal_count: number;
}

interface AttributionData {
  window_hours: number;
  total_decisions: number;
  signals: SignalStat[];
  asset_breakdown: AssetBreakdown[];
}

export default function SignalAttributionPanel() {
  const [data, setData] = useState<AttributionData | null>(null);

  useEffect(() => {
    const fetch = () =>
      api.signalAttribution(24).then(d => setData(d as unknown as AttributionData)).catch(() => {});
    fetch();
    const id = setInterval(fetch, 60_000);
    return () => clearInterval(id);
  }, []);

  if (!data || data.total_decisions === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Signal Attribution</h3>
        <div className="text-sm text-gray-600 py-4 text-center">No trade decisions yet</div>
      </div>
    );
  }

  // Top 10 signals by absolute impact
  const chartData = data.signals.slice(0, 10).map(s => ({
    name: s.signal.replace(/_/g, ' '),
    value: s.avg_contribution,
    absValue: s.abs_total,
    count: s.trade_count,
  }));

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Signal Attribution (24h)</h3>
        <span className="text-[10px] text-gray-600">{data.total_decisions} decisions</span>
      </div>

      {/* Bar chart of average signal contributions */}
      <div className="h-48 mb-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 10, bottom: 0, left: 80 }}>
            <XAxis type="number" tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false} />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              axisLine={false}
              tickLine={false}
              width={80}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a2235',
                border: '1px solid #243049',
                borderRadius: 8,
                fontSize: 11,
                color: '#e5e7eb',
              }}
              formatter={(value: number) => [value.toFixed(6), 'Avg Contribution']}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((d, i) => (
                <Cell key={i} fill={d.value >= 0 ? '#00d395' : '#ff4757'} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Signal table */}
      <div className="max-h-48 overflow-y-auto">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-surface-1">
            <tr className="text-gray-600 border-b border-surface-3">
              <th className="text-left py-1 font-medium">Signal</th>
              <th className="text-right py-1 font-medium">Trades</th>
              <th className="text-right py-1 font-medium">Avg</th>
              <th className="text-right py-1 font-medium">|Total|</th>
              <th className="text-right py-1 font-medium">B/S</th>
            </tr>
          </thead>
          <tbody>
            {data.signals.map((s, i) => (
              <tr key={i} className="border-b border-surface-3/30">
                <td className="py-1 text-gray-300 font-mono">{s.signal}</td>
                <td className="py-1 text-right text-gray-400 font-mono">{s.trade_count}</td>
                <td className={`py-1 text-right font-mono ${s.avg_contribution >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                  {s.avg_contribution.toFixed(4)}
                </td>
                <td className="py-1 text-right text-gray-400 font-mono">{s.abs_total.toFixed(2)}</td>
                <td className="py-1 text-right text-gray-500 font-mono">{s.buy_count}/{s.sell_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Asset breakdown */}
      {data.asset_breakdown.length > 0 && (
        <div className="mt-3 pt-3 border-t border-surface-3">
          <p className="text-[10px] text-gray-600 mb-2">Top signal per asset</p>
          <div className="flex flex-wrap gap-2">
            {data.asset_breakdown.map((a, i) => (
              <div key={i} className="text-[10px] px-2 py-1 bg-surface-2 rounded border border-surface-3">
                <span className="text-gray-400">{a.product_id}</span>
                <span className="mx-1 text-gray-600">&rarr;</span>
                <span className={a.top_signal_value >= 0 ? 'text-accent-green' : 'text-accent-red'}>
                  {a.top_signal}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
