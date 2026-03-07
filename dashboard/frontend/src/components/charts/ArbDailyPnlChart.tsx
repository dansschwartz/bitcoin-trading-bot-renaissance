import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { api } from '../../api';
import { formatCurrency } from '../../utils/formatters';

interface HourlyRow {
  hour: string;
  pnl: number;
  trades: number;
  wins: number;
}

export default function ArbDailyPnlChart() {
  const [data, setData] = useState<HourlyRow[]>([]);

  useEffect(() => {
    api.arbHourlyPnl(48).then(setData).catch(() => {});
    const id = setInterval(() => api.arbHourlyPnl(48).then(setData).catch(() => {}), 30_000);
    return () => clearInterval(id);
  }, []);

  if (!data.length) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Hourly P&L (48h)</h3>
        <div className="text-sm text-gray-600 py-8 text-center">No hourly data yet</div>
      </div>
    );
  }

  // Format hour for x-axis: "Mar 7 3pm"
  const formatted = data.map((d) => {
    const dt = new Date(d.hour.replace(' ', 'T') + ':00');
    const h = dt.getHours();
    const ampm = h >= 12 ? 'pm' : 'am';
    const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h;
    const label = `${dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} ${h12}${ampm}`;
    return { ...d, label };
  });

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Hourly P&L (48h)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={formatted} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="label"
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            axisLine={{ stroke: '#374151' }}
            tickLine={false}
            interval="preserveStartEnd"
            angle={-35}
            textAnchor="end"
            height={50}
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
            formatter={(value: number, _name: string, props: { payload?: HourlyRow }) => {
              const trades = props.payload?.trades ?? 0;
              const wins = props.payload?.wins ?? 0;
              return [
                `${formatCurrency(value)}  (${trades} trades, ${wins} wins)`,
                'P&L',
              ];
            }}
          />
          <ReferenceLine y={0} stroke="#374151" />
          <Bar dataKey="pnl" radius={[4, 4, 0, 0]} maxBarSize={20}>
            {formatted.map((entry, idx) => (
              <Cell key={idx} fill={entry.pnl >= 0 ? '#00d395' : '#ff4757'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
