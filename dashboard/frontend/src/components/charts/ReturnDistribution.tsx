import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { api } from '../../api';

interface Bucket {
  range: string;
  count: number;
  isPositive: boolean;
}

export default function ReturnDistribution() {
  const [buckets, setBuckets] = useState<Bucket[]>([]);

  useEffect(() => {
    api.distribution().then((raw) => {
      const values = raw.map(r => r.trade_pnl).filter(v => v !== 0);
      if (values.length === 0) return;

      // Create histogram buckets
      const min = Math.min(...values);
      const max = Math.max(...values);
      const nBuckets = 12;
      const step = (max - min) / nBuckets || 1;

      const hist: Bucket[] = [];
      for (let i = 0; i < nBuckets; i++) {
        const lo = min + i * step;
        const hi = lo + step;
        const count = values.filter(v => v >= lo && (i === nBuckets - 1 ? v <= hi : v < hi)).length;
        const mid = (lo + hi) / 2;
        hist.push({
          range: `${lo >= 0 ? '+' : ''}${lo.toFixed(0)}`,
          count,
          isPositive: mid >= 0,
        });
      }
      setBuckets(hist);
    }).catch(() => {});
  }, []);

  if (buckets.length === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Return Distribution</h3>
        <div className="h-48 flex items-center justify-center text-gray-600 text-sm">No data</div>
      </div>
    );
  }

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Return Distribution</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={buckets} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <XAxis dataKey="range" tick={{ fontSize: 9, fill: '#6b7280' }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #243049', borderRadius: 8, fontSize: 12 }} />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {buckets.map((b, i) => (
                <Cell key={i} fill={b.isPositive ? '#00d395' : '#ff4757'} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
