import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import { api } from '../../api';
import type { MLPrediction } from '../../types';
import { CHART_COLORS } from '../../utils/colors';
import { formatTimestamp } from '../../utils/formatters';

/** Round an ISO timestamp string to the nearest minute for cross-model grouping. */
function roundToMinute(ts: string): string {
  const d = new Date(ts);
  d.setSeconds(0, 0);
  return d.toISOString();
}

/** Downsample an array to at most `maxPoints` evenly-spaced entries. */
function downsample<T>(arr: T[], maxPoints: number): T[] {
  if (arr.length <= maxPoints) return arr;
  const step = arr.length / maxPoints;
  const result: T[] = [];
  for (let i = 0; i < maxPoints; i++) {
    result.push(arr[Math.floor(i * step)]);
  }
  return result;
}

export default function PredictionHistory() {
  const [data, setData] = useState<Record<string, number | string>[]>([]);
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    const load = () => api.predictionHistory(24).then((preds) => {
      // Group by rounded timestamp so models with slightly different ms align
      const modelSet = new Set<string>();
      const byTime = new Map<string, Record<string, number | string>>();

      for (const p of preds) {
        modelSet.add(p.model_name);
        const ts = roundToMinute(p.timestamp);
        if (!byTime.has(ts)) byTime.set(ts, { timestamp: ts });
        // Use latest prediction for this model at this rounded timestamp
        byTime.get(ts)![p.model_name] = p.prediction;
      }

      setModels(Array.from(modelSet));
      // Reverse for chronological order, then downsample for Recharts performance
      const chronological = Array.from(byTime.values()).reverse();
      setData(downsample(chronological, 200));
    }).catch(() => {});
    load();
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  }, []);

  if (data.length === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">ML Prediction History</h3>
        <div className="h-48 flex flex-col items-center justify-center text-gray-600 text-sm gap-2">
          <span>Collecting predictions...</span>
          <span className="text-xs text-gray-700">ML models log predictions each trading cycle</span>
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
            <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #243049', borderRadius: 8, fontSize: 12, color: '#e5e7eb' }}
              labelFormatter={formatTimestamp}
            />
            <ReferenceLine y={0} stroke="#374151" strokeDasharray="3 3" />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {models.map((m, i) => (
              <Line key={m} type="monotone" dataKey={m} stroke={CHART_COLORS[i % CHART_COLORS.length]} dot={false} strokeWidth={1.5} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[10px] text-gray-600 mt-1">{data.length} data points (downsampled from 24h)</p>
    </div>
  );
}
