import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { api } from '../../api';
import type { EnsembleStatus } from '../../types';
import { CHART_COLORS } from '../../utils/colors';

export default function EnsemblePanel() {
  const [data, setData] = useState<EnsembleStatus | null>(null);

  useEffect(() => {
    api.ensemble().then(setData).catch(() => {});
    const id = setInterval(() => api.ensemble().then(setData).catch(() => {}), 15_000);
    return () => clearInterval(id);
  }, []);

  if (!data || data.model_count === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">ML Ensemble</h3>
        <div className="h-32 flex items-center justify-center text-gray-600 text-sm">
          No ML predictions available
        </div>
      </div>
    );
  }

  const chartData = Object.entries(data.models).map(([name, pred]) => ({
    name,
    prediction: pred.prediction,
  }));

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">ML Ensemble</h3>
        <span className="text-xs text-gray-500">{data.model_count} models</span>
      </div>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ left: 60, right: 10, top: 5, bottom: 5 }}>
            <XAxis type="number" domain={['auto', 'auto']} tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #243049', borderRadius: 8, fontSize: 12, color: '#e5e7eb' }} />
            <Bar dataKey="prediction" radius={[0, 4, 4, 0]}>
              {chartData.map((d, i) => (
                <Cell key={i} fill={d.prediction >= 0 ? '#00d395' : '#ff4757'} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
