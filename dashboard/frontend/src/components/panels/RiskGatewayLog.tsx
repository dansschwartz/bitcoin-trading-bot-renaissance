import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { RiskGatewayEntry } from '../../types';
import { formatTimestamp } from '../../utils/formatters';

export default function RiskGatewayLog() {
  const [entries, setEntries] = useState<RiskGatewayEntry[]>([]);

  useEffect(() => {
    api.gatewayLog(50).then(setEntries).catch(() => {});
    const id = setInterval(() => api.gatewayLog(50).then(setEntries).catch(() => {}), 15_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Risk Gateway Log</h3>
      <div className="overflow-y-auto max-h-64">
        <table className="w-full text-xs font-mono">
          <thead className="sticky top-0 bg-surface-1">
            <tr className="text-gray-500 border-b border-surface-3">
              <th className="text-left py-1.5 px-2">Time</th>
              <th className="text-left py-1.5 px-2">Action</th>
              <th className="text-right py-1.5 px-2">Confidence</th>
              <th className="text-right py-1.5 px-2">VAE Loss</th>
              <th className="text-left py-1.5 px-2">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => {
              const verdict = e.gateway_verdict || ((e.vae_loss || 0) < 0.3 ? 'PASS' : 'BLOCK');
              const pass = verdict === 'PASS';
              return (
                <tr key={e.id} className="border-b border-surface-3/30">
                  <td className="py-1.5 px-2 text-gray-500">{formatTimestamp(e.timestamp)}</td>
                  <td className="py-1.5 px-2 text-gray-300">{e.action}</td>
                  <td className="py-1.5 px-2 text-right text-gray-400">{(e.confidence * 100).toFixed(0)}%</td>
                  <td className="py-1.5 px-2 text-right text-gray-400">{e.vae_loss != null ? e.vae_loss.toFixed(4) : '--'}</td>
                  <td className="py-1.5 px-2">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                      pass ? 'bg-accent-green/20 text-accent-green' : 'bg-accent-red/20 text-accent-red'
                    }`}>
                      {verdict}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
