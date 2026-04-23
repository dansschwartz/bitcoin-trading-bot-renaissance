import { useEffect, useState } from 'react';
import { api } from '../../api';

interface Check {
  id: string;
  label: string;
  passed: boolean;
}

interface CriteriaData {
  checks: Check[];
  passed: number;
  total: number;
}

export default function SuccessCriteriaPanel() {
  const [data, setData] = useState<CriteriaData | null>(null);

  useEffect(() => {
    const fetch = () => {
      api.successCriteria().then(setData).catch(() => {});
    };
    fetch();
    const id = setInterval(fetch, 30_000);
    return () => clearInterval(id);
  }, []);

  if (!data) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <p className="text-xs text-gray-500 uppercase tracking-wider">System Health</p>
        <p className="text-sm text-gray-600 mt-2">Loading...</p>
      </div>
    );
  }

  const pct = data.total > 0 ? Math.round((data.passed / data.total) * 100) : 0;
  const barColor = pct >= 80 ? 'bg-accent-green' : pct >= 50 ? 'bg-yellow-400' : 'bg-accent-red';
  const textColor = pct >= 80 ? 'text-accent-green' : pct >= 50 ? 'text-yellow-400' : 'text-accent-red';

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-500 uppercase tracking-wider">Success Criteria</p>
        <span className={`text-lg font-semibold font-mono ${textColor}`}>
          {data.passed}/{data.total}
        </span>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-surface-2 rounded-full h-2">
        <div
          className={`${barColor} h-2 rounded-full transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Checklist */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
        {data.checks.map(check => (
          <div key={check.id} className="flex items-center gap-2 py-0.5">
            <span className={`text-xs ${check.passed ? 'text-accent-green' : 'text-gray-600'}`}>
              {check.passed ? '\u2713' : '\u2717'}
            </span>
            <span className={`text-[11px] ${check.passed ? 'text-gray-300' : 'text-gray-500'}`}>
              {check.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
