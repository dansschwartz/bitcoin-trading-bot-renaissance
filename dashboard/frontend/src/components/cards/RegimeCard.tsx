import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { RegimeStatus } from '../../types';
import { regimeColor } from '../../utils/colors';

interface LiveRegime {
  hmm_regime?: string;
  confidence?: number;
  classifier?: string;
  bar_count?: number;
  details?: string;
  weighted_signal?: number;
}

export default function RegimeCard() {
  const [data, setData] = useState<RegimeStatus | null>(null);

  useEffect(() => {
    api.regime().then(setData).catch(() => {});
    const id = setInterval(() => api.regime().then(setData).catch(() => {}), 15_000);
    return () => clearInterval(id);
  }, []);

  const current = data?.current as LiveRegime | null;
  const regime = current?.hmm_regime || 'Unknown';
  const color = regimeColor(regime);
  const classifier = current?.classifier || 'none';
  const barCount = current?.bar_count ?? 0;
  const confidence = current?.confidence ?? 0;

  const classifierLabel = classifier === 'hmm' ? 'HMM' : classifier === 'bootstrap' ? 'Bootstrap' : 'Waiting';
  const classifierColor = classifier === 'hmm' ? 'text-emerald-400' : classifier === 'bootstrap' ? 'text-amber-400' : 'text-gray-500';
  const barsNeeded = classifier === 'bootstrap' ? `${barCount}/200 bars` : classifier === 'hmm' ? `${barCount} bars` : `${barCount}/20 bars`;

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-500 uppercase tracking-wider">Market Regime</p>
        <span className={`text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded ${classifierColor} bg-surface-2`}>
          {classifierLabel}
        </span>
      </div>
      <div className="flex items-center gap-2 mt-2">
        <span
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: color }}
        />
        <span className="text-lg font-semibold font-mono capitalize" style={{ color }}>
          {regime}
        </span>
      </div>
      <div className="flex items-center justify-between mt-1">
        <p className="text-xs text-gray-500">
          Confidence: {(confidence * 100).toFixed(1)}%
        </p>
        <p className="text-[10px] text-gray-600 font-mono">{barsNeeded}</p>
      </div>
      {current?.details && (
        <p className="text-[10px] text-gray-600 mt-0.5 font-mono truncate">{current.details}</p>
      )}
    </div>
  );
}
