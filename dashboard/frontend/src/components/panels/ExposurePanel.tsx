import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { Exposure } from '../../types';
import { formatCurrency } from '../../utils/formatters';
import Gauge from '../shared/Gauge';

export default function ExposurePanel() {
  const [data, setData] = useState<Exposure | null>(null);

  useEffect(() => {
    api.exposure().then(setData).catch(() => {});
    const id = setInterval(() => api.exposure().then(setData).catch(() => {}), 10_000);
    return () => clearInterval(id);
  }, []);

  if (!data) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Exposure</h3>
        <div className="text-sm text-gray-600 py-4 text-center">Loading...</div>
      </div>
    );
  }

  // Normalize gross exposure against 1000 (position limit)
  const utilization = Math.min(data.gross_exposure / 1000, 1);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Exposure</h3>
      <div className="flex items-start gap-6">
        <Gauge
          value={utilization}
          label="Utilization"
          color={utilization > 0.8 ? '#ff4757' : utilization > 0.5 ? '#fbbf24' : '#00d395'}
          size={80}
        />
        <div className="flex-1 space-y-2 text-xs font-mono">
          <div className="flex justify-between">
            <span className="text-gray-500">Long</span>
            <span className="text-accent-green">{formatCurrency(data.long_exposure)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Short</span>
            <span className="text-accent-red">{formatCurrency(data.short_exposure)}</span>
          </div>
          <div className="flex justify-between border-t border-surface-3 pt-1">
            <span className="text-gray-500">Net</span>
            <span className="text-gray-300">{formatCurrency(data.net_exposure)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Gross</span>
            <span className="text-gray-300">{formatCurrency(data.gross_exposure)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Positions</span>
            <span className="text-gray-300">{data.position_count}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
