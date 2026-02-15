import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { api } from '../../api';
import type { MLPrediction } from '../../types';
import { CHART_COLORS } from '../../utils/colors';
import { formatTimestamp } from '../../utils/formatters';

export default function PredictionHistory() {
  const [data, setData] = useState<Record<string, number | string>[]>([]);
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    api.predictionHistory(24).then((preds) => {
      // Group by timestamp, pivot model names into columns
      const modelSet = new Set<string>();
      const byTime = new Map<string, Record<string, number | string>>();

      for (const p of preds) {
        modelSet.add(p.model_name);
        const ts = p.timestamp;
        if (!byTime.has(ts)) byTime.set(ts, { timestamp: ts });
        byTime.get(ts)![p.model_name] = p.prediction;
      }

      setModels(Array.from(modelSet));
      setData(Array.from(byTime.values()).reverse());
    }).catch(() => {});
  }, []);

  if (data.length === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">ML Prediction History</h3>
        <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
          No predictions yet
        </div>
      </div>
    );
  }

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">ML Prediction History (24h)</h3>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <XAxis dataKey="timestamp" tickFormatter={formatTimestamp} tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false} domain={[-1, 1]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #243049', borderRadius: 8, fontSize: 12 }}
              labelFormatter={formatTimestamp}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {models.map((m, i) => (
              <Line key={m} type="monotone" dataKey={m} stroke={CHART_COLORS[i % CHART_COLORS.length]} dot={false} strokeWidth={1.5} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
