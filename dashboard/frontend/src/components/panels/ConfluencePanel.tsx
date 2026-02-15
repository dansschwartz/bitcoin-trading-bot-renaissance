import { useEffect, useState } from 'react';
import { api } from '../../api';

export default function ConfluencePanel() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    api.confluence().then(setData).catch(() => {});
    const id = setInterval(() => api.confluence().then(setData).catch(() => {}), 15_000);
    return () => clearInterval(id);
  }, []);

  if (!data || data.status === 'no_live_data') {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Confluence Engine</h3>
        <div className="text-sm text-gray-600 py-4 text-center">
          Waiting for live bot cycle data...
        </div>
      </div>
    );
  }

  const boost = Number(data.total_confluence_boost || 0);
  const rules = (data.active_rules as Record<string, unknown>[]) || [];

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Confluence Engine</h3>
        <span className={`text-xs font-mono font-semibold ${boost > 0 ? 'text-accent-green' : 'text-gray-500'}`}>
          Boost: {boost > 0 ? '+' : ''}{(boost * 100).toFixed(1)}%
        </span>
      </div>
      {rules.length > 0 ? (
        <div className="space-y-1">
          {rules.map((rule, i) => (
            <div key={i} className="flex items-center justify-between text-xs py-1 border-b border-surface-3/30">
              <span className="text-gray-400">{String(rule.name || `Rule ${i + 1}`)}</span>
              <span className="text-accent-green font-mono">
                +{(Number(rule.boost || 0) * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-gray-600">No active confluence rules</p>
      )}
    </div>
  );
}
