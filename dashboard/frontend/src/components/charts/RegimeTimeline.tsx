import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { Decision } from '../../types';
import { regimeColor } from '../../utils/colors';

export default function RegimeTimeline() {
  const [history, setHistory] = useState<Decision[]>([]);

  useEffect(() => {
    api.regime().then(r => setHistory((r.history || []).slice(0, 100).reverse())).catch(() => {});
  }, []);

  if (history.length === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Regime Timeline</h3>
        <div className="h-12 flex items-center justify-center text-gray-600 text-sm">No regime data</div>
      </div>
    );
  }

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Regime Timeline</h3>
      <div className="flex gap-px h-8 rounded overflow-hidden">
        {history.map((d, i) => (
          <div
            key={i}
            className="flex-1 min-w-[3px] cursor-default"
            style={{ backgroundColor: regimeColor(d.hmm_regime) }}
            title={`${d.hmm_regime} | ${d.timestamp}`}
          />
        ))}
      </div>
      {/* Legend */}
      <div className="flex gap-4 mt-2">
        {Array.from(new Set(history.map(d => d.hmm_regime).filter(Boolean))).map(r => (
          <div key={r} className="flex items-center gap-1.5 text-[10px] text-gray-500">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: regimeColor(r!) }} />
            <span className="capitalize">{r}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
