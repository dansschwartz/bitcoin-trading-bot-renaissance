import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { CalendarPnL } from '../../types';
import { formatCurrency } from '../../utils/formatters';
import { pnlBgColor } from '../../utils/colors';

export default function CalendarHeatmap() {
  const [data, setData] = useState<CalendarPnL[]>([]);

  useEffect(() => {
    api.calendar().then(setData).catch(() => {});
  }, []);

  if (data.length === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Daily P&L Calendar</h3>
        <div className="h-32 flex items-center justify-center text-gray-600 text-sm">No data</div>
      </div>
    );
  }

  const maxAbs = Math.max(...data.map(d => Math.abs(d.daily_pnl)), 1);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Daily P&L Calendar</h3>
      <div className="grid grid-cols-7 gap-1">
        {data.slice(0, 42).reverse().map((d) => {
          const intensity = Math.min(Math.abs(d.daily_pnl) / maxAbs, 1);
          const color = d.daily_pnl >= 0
            ? `rgba(0, 211, 149, ${0.15 + intensity * 0.6})`
            : `rgba(255, 71, 87, ${0.15 + intensity * 0.6})`;
          return (
            <div
              key={d.date}
              className="aspect-square rounded-sm flex items-center justify-center text-[8px] font-mono cursor-default"
              style={{ backgroundColor: color }}
              title={`${d.date}: ${formatCurrency(d.daily_pnl)} (${d.trade_count} trades)`}
            >
              {new Date(d.date).getDate()}
            </div>
          );
        })}
      </div>
    </div>
  );
}
