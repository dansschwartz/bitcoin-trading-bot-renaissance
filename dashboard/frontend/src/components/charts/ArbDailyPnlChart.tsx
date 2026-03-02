import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { api } from '../../api';
import type { ArbDailyPnl } from '../../types';
import { formatCurrency } from '../../utils/formatters';

export default function ArbDailyPnlChart() {
  const [data, setData] = useState<ArbDailyPnl[]>([]);

  useEffect(() => {
    api.arbDailyPnl(21).then(setData).catch(() => {});
    const id = setInterval(() => api.arbDailyPnl(21).then(setData).catch(() => {}), 30_000);
    return () => clearInterval(id);
  }, []);

  if (!data.length) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Daily P&L (21 days)</h3>
        <div className="text-sm text-gray-600 py-8 text-center">No daily data yet</div>
      </div>
    );
  }

  // Format date for x-axis: "Feb 28"
  const formatted = data.map((d) => ({
    ...d,
    label: new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
  }));

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Daily P&L (21 days)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={formatted} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="label"
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            axisLine={{ stroke: '#374151' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#6b7280', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            width={50}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: 8 }}
            labelStyle={{ color: '#9ca3af', fontSize: 12 }}
            itemStyle={{ color: '#e5e7eb', fontSize: 12 }}
            formatter={(value: number, _name: string, props: { payload?: ArbDailyPnl }) => {
              const trades = props.payload?.trades ?? 0;
              const wins = props.payload?.wins ?? 0;
              return [
                `${formatCurrency(value)}  (${trades} trades, ${wins} wins)`,
                'P&L',
              ];
            }}
          />
          <ReferenceLine y={0} stroke="#374151" />
          <Bar dataKey="pnl" radius={[4, 4, 0, 0]} maxBarSize={36}>
            {formatted.map((entry, idx) => (
              <Cell key={idx} fill={entry.pnl >= 0 ? '#00d395' : '#ff4757'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
