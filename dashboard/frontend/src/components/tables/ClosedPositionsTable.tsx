import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { Trade } from '../../types';
import { formatCurrency, formatTimestamp } from '../../utils/formatters';

export default function ClosedPositionsTable() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [offset, setOffset] = useState(0);
  const limit = 25;

  useEffect(() => {
    api.closedTrades(limit, offset).then(setTrades).catch(() => {});
  }, [offset]);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Trade History</h3>
        <div className="flex items-center gap-2 text-xs">
          <button
            onClick={() => setOffset(Math.max(0, offset - limit))}
            disabled={offset === 0}
            className="px-2 py-1 bg-surface-2 rounded text-gray-400 hover:text-gray-200 disabled:opacity-30"
          >
            Prev
          </button>
          <span className="text-gray-500">{offset + 1}-{offset + trades.length}</span>
          <button
            onClick={() => setOffset(offset + limit)}
            disabled={trades.length < limit}
            className="px-2 py-1 bg-surface-2 rounded text-gray-400 hover:text-gray-200 disabled:opacity-30"
          >
            Next
          </button>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-gray-500 border-b border-surface-3">
              <th className="text-left py-2 px-2">Time</th>
              <th className="text-left py-2 px-2">Asset</th>
              <th className="text-left py-2 px-2">Side</th>
              <th className="text-right py-2 px-2">Size</th>
              <th className="text-right py-2 px-2">Price</th>
              <th className="text-left py-2 px-2">Algo</th>
              <th className="text-right py-2 px-2">Slippage</th>
              <th className="text-left py-2 px-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={t.id} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                <td className="py-2 px-2 text-gray-500">{formatTimestamp(t.timestamp)}</td>
                <td className="py-2 px-2 text-gray-300">{t.product_id}</td>
                <td className="py-2 px-2">
                  <span className={t.side === 'BUY' ? 'text-accent-green' : 'text-accent-red'}>
                    {t.side}
                  </span>
                </td>
                <td className="py-2 px-2 text-right text-gray-300">{t.size.toFixed(6)}</td>
                <td className="py-2 px-2 text-right text-gray-300">{formatCurrency(t.price)}</td>
                <td className="py-2 px-2 text-gray-500">{t.algo_used || '--'}</td>
                <td className="py-2 px-2 text-right text-gray-500">
                  {t.slippage != null ? `${(t.slippage * 100).toFixed(3)}%` : '--'}
                </td>
                <td className="py-2 px-2">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                    t.status === 'FILLED' ? 'bg-accent-green/20 text-accent-green' : 'bg-surface-3 text-gray-400'
                  }`}>
                    {t.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
