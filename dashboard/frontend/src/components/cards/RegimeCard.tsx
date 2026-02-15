import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { RegimeStatus } from '../../types';
import { regimeColor } from '../../utils/colors';

export default function RegimeCard() {
  const [data, setData] = useState<RegimeStatus | null>(null);

  useEffect(() => {
    api.regime().then(setData).catch(() => {});
    const id = setInterval(() => api.regime().then(setData).catch(() => {}), 15_000);
    return () => clearInterval(id);
  }, []);

  const current = data?.current;
  const regime = current?.hmm_regime || 'Unknown';
  const color = regimeColor(regime);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wider">HMM Regime</p>
      <div className="flex items-center gap-2 mt-2">
        <span
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: color }}
        />
        <span className="text-lg font-semibold font-mono capitalize" style={{ color }}>
          {regime}
        </span>
      </div>
      {current && (
        <p className="text-xs text-gray-500 mt-1">
          Confidence: {(current.confidence * 100).toFixed(1)}% | Signal: {current.weighted_signal?.toFixed(4)}
        </p>
      )}
    </div>
  );
}
