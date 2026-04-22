import { useEffect, useState } from 'react';
import { api } from '../../api';

interface EvaluatedRule {
  name: string;
  signals: string[];
  boost: number;
  fired: boolean;
}

interface ConfluenceData {
  total_confluence_boost: number;
  active_rules: { name: string; boost: number }[];
  evaluated_rules?: EvaluatedRule[];
  signal_count?: number;
  status?: string;
  timestamp?: string;
}

export default function ConfluencePanel() {
  const [data, setData] = useState<ConfluenceData | null>(null);

  useEffect(() => {
    const fetch = () => api.confluence().then(d => setData(d as unknown as ConfluenceData)).catch(() => {});
    fetch();
    const id = setInterval(fetch, 15_000);
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
  const rules = data.evaluated_rules || [];
  const activeCount = rules.filter(r => r.fired).length;

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Confluence Engine</h3>
        <div className="flex items-center gap-2">
          {data.signal_count != null && (
            <span className="text-[10px] text-gray-600">{data.signal_count} signals</span>
          )}
          <span className={`text-xs font-mono font-semibold ${boost > 0 ? 'text-accent-green' : 'text-gray-500'}`}>
            Boost: {boost > 0 ? '+' : ''}{(boost * 100).toFixed(1)}%
          </span>
        </div>
      </div>
      {rules.length > 0 ? (
        <div className="space-y-1">
          {rules.map((rule, i) => (
            <div key={i} className={`flex items-center justify-between text-xs py-1.5 px-2 rounded ${
              rule.fired ? 'bg-accent-green/10' : 'bg-surface-2/30'
            }`}>
              <div className="flex items-center gap-2 min-w-0">
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                  rule.fired ? 'bg-accent-green' : 'bg-gray-700'
                }`} />
                <span className={rule.fired ? 'text-gray-200' : 'text-gray-600'}>{rule.name}</span>
              </div>
              <span className={`font-mono shrink-0 ${rule.fired ? 'text-accent-green' : 'text-gray-700'}`}>
                {rule.fired ? `+${(rule.boost * 100).toFixed(1)}%` : 'inactive'}
              </span>
            </div>
          ))}
          <p className="text-[10px] text-gray-600 mt-1">
            {activeCount}/{rules.length} rules active
          </p>
        </div>
      ) : (
        <p className="text-xs text-gray-600">No confluence rules evaluated</p>
      )}
    </div>
  );
}
