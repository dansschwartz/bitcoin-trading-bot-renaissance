import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { api } from '../../api';
import type { VAEPoint } from '../../types';
import { formatTimestamp } from '../../utils/formatters';
import Gauge from '../shared/Gauge';

export default function VAEGauge() {
  const [data, setData] = useState<VAEPoint[]>([]);

  useEffect(() => {
    api.vae().then(d => setData(d.slice(0, 100).reverse())).catch(() => {});
    const id = setInterval(() => api.vae().then(d => setData(d.slice(0, 100).reverse())).catch(() => {}), 30_000);
    return () => clearInterval(id);
  }, []);

  const latest = data.length > 0 ? data[data.length - 1].vae_loss : null;
  const ANOMALY_THRESHOLD = 5.0;
  // Normalize VAE loss: 0 → 0, threshold → 1.0
  const normalized = latest != null ? Math.min(latest / ANOMALY_THRESHOLD, 1) : 0;
  const gaugeColor = normalized > 0.6 ? '#ff4757' : normalized > 0.3 ? '#fbbf24' : '#00d395';

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">VAE Anomaly Detector</h3>
      {data.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-32 text-center">
          <Gauge value={0} label="Anomaly Score" color="#374151" size={70} />
          <p className="text-xs text-gray-600 mt-2">Collecting data — VAE scores appear after trades execute</p>
        </div>
      ) : (
        <>
          <div className="flex items-start gap-4">
            <Gauge value={normalized} label="Anomaly Score" color={gaugeColor} size={90} />
            <div className="flex-1 h-24">
              {data.length > 1 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data} margin={{ top: 2, right: 5, bottom: 2, left: 5 }}>
                    <XAxis dataKey="timestamp" tickFormatter={formatTimestamp} tick={{ fontSize: 8, fill: '#6b7280' }} axisLine={false} tickLine={false} hide />
                    <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} axisLine={false} tickLine={false} width={30} />
                    <Tooltip contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #243049', borderRadius: 8, fontSize: 11, color: '#e5e7eb' }} labelFormatter={formatTimestamp} />
                    <ReferenceLine y={ANOMALY_THRESHOLD} stroke="#fbbf24" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="vae_loss" stroke="#a855f7" dot={false} strokeWidth={1.5} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-600 text-xs">Building history...</div>
              )}
            </div>
          </div>
          <p className="text-[10px] text-gray-600 mt-2">
            Latest reconstruction error: {latest != null ? latest.toFixed(4) : '--'} | Threshold: {ANOMALY_THRESHOLD}
          </p>
        </>
      )}
    </div>
  );
}
