import { useEffect, useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { api } from '../../api';
import type { EquityPoint, TimeRange } from '../../types';
import TimeRangeSelector from '../shared/TimeRangeSelector';
import { formatCurrency, formatTimestamp } from '../../utils/formatters';

export default function EquityCurve() {
  const [range, setRange] = useState<TimeRange>('1D');
  const [data, setData] = useState<EquityPoint[]>([]);

  useEffect(() => {
    api.equity(range).then(setData).catch(() => {});
  }, [range]);

  const isPositive = data.length > 0 && data[data.length - 1]?.cumulative_pnl >= 0;

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Equity Curve</h3>
        <TimeRangeSelector value={range} onChange={setRange} />
      </div>
      <div className="h-56">
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
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
                contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #243049', borderRadius: 8, fontSize: 12 }}
                labelFormatter={formatTimestamp}
                formatter={(value: number) => [formatCurrency(value), 'Cumulative P&L']}
              />
              <ReferenceLine y={0} stroke="#374151" strokeDasharray="3 3" />
              <Area
                type="monotone"
                dataKey="cumulative_pnl"
                stroke={isPositive ? '#00d395' : '#ff4757'}
                fill="url(#eqGrad)"
                strokeWidth={2}
              />
            </AreaChart>
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
