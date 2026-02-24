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

  // Normalize gross exposure against 10000 (capital)
  const utilization = Math.min(data.gross_exposure / 10000, 1);
  const assets = Object.entries(data.positions_by_asset || {});

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
            <span className="text-gray-500" title="Long - Short (directional bias)">Net (L-S)</span>
            <span className={data.net_exposure >= 0 ? 'text-accent-green' : 'text-accent-red'}>
              {formatCurrency(data.net_exposure)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500" title="Long + Short (total capital deployed)">Gross (L+S)</span>
            <span className="text-gray-300">{formatCurrency(data.gross_exposure)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Positions</span>
            <span className="text-gray-300">{data.position_count}</span>
          </div>
        </div>
      </div>

      {/* Per-asset breakdown */}
      {assets.length > 0 && (
        <div className="mt-3 border-t border-surface-3 pt-2">
          <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5">By Asset (Netted)</p>
          <div className="space-y-1">
            {assets.map(([asset, info]) => (
              <div key={asset} className="flex justify-between text-xs font-mono">
                <span className="text-gray-400">{asset}</span>
                <span className="text-gray-500">{info.count} pos</span>
                <span className={info.net_value >= 0 ? 'text-accent-green' : 'text-accent-red'}>
                  {formatCurrency(info.net_value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
