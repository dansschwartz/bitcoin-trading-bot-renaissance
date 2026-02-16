import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { Exposure } from '../../types';
import { formatCurrency } from '../../utils/formatters';

interface Props {
  compact?: boolean;
}

export default function AssetSummaryPanel({ compact = false }: Props) {
  const [data, setData] = useState<Exposure | null>(null);

  useEffect(() => {
    api.exposure().then(setData).catch(() => {});
    const id = setInterval(() => api.exposure().then(setData).catch(() => {}), 10_000);
    return () => clearInterval(id);
  }, []);

  if (!data) return null;
  const assets = Object.entries(data.positions_by_asset || {});

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">
          {compact ? 'Exposure' : 'Asset Summary (Netted)'}
        </h3>
        <div className="flex items-center gap-3 text-[10px] font-mono">
          <span className="text-accent-green">L: {formatCurrency(data.long_exposure)}</span>
          <span className="text-accent-red">S: {formatCurrency(data.short_exposure)}</span>
          <span className={`font-semibold ${data.net_exposure >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            Net: {formatCurrency(data.net_exposure)}
          </span>
        </div>
      </div>
      {assets.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="text-left py-1.5 px-2">Asset</th>
                <th className="text-right py-1.5 px-2">#</th>
                <th className="text-right py-1.5 px-2">Long</th>
                <th className="text-right py-1.5 px-2">Short</th>
                <th className="text-right py-1.5 px-2">Net</th>
              </tr>
            </thead>
            <tbody>
              {assets.map(([asset, info]) => (
                <tr key={asset} className="border-b border-surface-3/50">
                  <td className="py-1.5 px-2 text-gray-300">{asset.replace('-USD', '')}</td>
                  <td className="py-1.5 px-2 text-right text-gray-400">{info.count}</td>
                  <td className="py-1.5 px-2 text-right text-accent-green">
                    {info.long_value > 0 ? formatCurrency(info.long_value) : '—'}
                  </td>
                  <td className="py-1.5 px-2 text-right text-accent-red">
                    {info.short_value > 0 ? formatCurrency(info.short_value) : '—'}
                  </td>
                  <td className={`py-1.5 px-2 text-right font-semibold ${info.net_value >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                    {formatCurrency(info.net_value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center text-gray-600 text-xs py-4">No open positions</div>
      )}
    </div>
  );
}
