import { useEffect, useState } from 'react';
import { api } from '../../api';
import { formatTimestamp } from '../../utils/formatters';

interface DevilEntry {
  pair: string;
  side: string;
  signal_price: number;
  fill_price: number;
  slippage_bps: number;
  fill_fee: number;
  devil: number;
  latency_signal_to_fill_ms: number;
  signal_timestamp: string;
}

interface DevilSummary {
  total_signals: number;
  total_fills: number;
  avg_slippage_bps: number;
  avg_latency_ms: number;
  total_devil_cost_usd: number;
  total_fees_usd: number;
  recent: DevilEntry[];
}

export default function DevilTrackerPanel() {
  const [data, setData] = useState<DevilSummary | null>(null);

  useEffect(() => {
    const fetch = () =>
      api.devilSummary(24).then(d => setData(d as unknown as DevilSummary)).catch(() => {});
    fetch();
    const id = setInterval(fetch, 30_000);
    return () => clearInterval(id);
  }, []);

  if (!data || data.total_signals === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Cost Leakage (The Devil)</h3>
        <div className="text-sm text-gray-600 py-4 text-center">No devil tracker data yet</div>
      </div>
    );
  }

  const costRatio = data.total_devil_cost_usd;
  const ratioColor =
    costRatio > 50 ? 'text-accent-red' : costRatio > 10 ? 'text-yellow-400' : 'text-accent-green';

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Cost Leakage (The Devil)</h3>
        <span className="text-[10px] text-gray-600">{data.total_signals} signals (24h)</span>
      </div>

      {/* Summary metrics */}
      <div className="grid grid-cols-4 gap-2 mb-3">
        <div className="text-center">
          <p className={`text-lg font-mono font-semibold ${ratioColor}`}>
            ${data.total_devil_cost_usd.toFixed(2)}
          </p>
          <p className="text-[10px] text-gray-600">Total Leakage</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-mono font-semibold text-gray-200">
            {data.avg_slippage_bps.toFixed(1)}
          </p>
          <p className="text-[10px] text-gray-600">Avg Slip (bps)</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-mono font-semibold text-gray-200">
            {data.avg_latency_ms.toFixed(1)}
          </p>
          <p className="text-[10px] text-gray-600">Avg Latency (ms)</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-mono font-semibold text-gray-200">
            ${data.total_fees_usd.toFixed(2)}
          </p>
          <p className="text-[10px] text-gray-600">Total Fees</p>
        </div>
      </div>

      {/* Recent entries table */}
      <div className="max-h-48 overflow-y-auto">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-surface-1">
            <tr className="text-gray-600 border-b border-surface-3">
              <th className="text-left py-1 font-medium">Pair</th>
              <th className="text-left py-1 font-medium">Side</th>
              <th className="text-right py-1 font-medium">Slip (bps)</th>
              <th className="text-right py-1 font-medium">Latency</th>
              <th className="text-right py-1 font-medium">Time</th>
            </tr>
          </thead>
          <tbody>
            {data.recent.slice(0, 10).map((entry, i) => (
              <tr key={i} className="border-b border-surface-3/30">
                <td className="py-1 text-gray-300 font-mono">{entry.pair}</td>
                <td className={`py-1 font-mono ${entry.side === 'BUY' ? 'text-accent-green' : 'text-accent-red'}`}>
                  {entry.side}
                </td>
                <td className="py-1 text-right text-gray-400 font-mono">
                  {entry.slippage_bps.toFixed(1)}
                </td>
                <td className="py-1 text-right text-gray-400 font-mono">
                  {entry.latency_signal_to_fill_ms.toFixed(1)}ms
                </td>
                <td className="py-1 text-right text-gray-600">{formatTimestamp(entry.signal_timestamp)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
