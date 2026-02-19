import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { ClosedPosition } from '../../types';
import { formatCurrency, formatTimestamp, pnlColor, pnlSign } from '../../utils/formatters';

function formatDuration(seconds: number | null): string {
  if (seconds == null) return '--';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}

export default function ClosedPositionsTable() {
  const [positions, setPositions] = useState<ClosedPosition[]>([]);
  const [offset, setOffset] = useState(0);
  const limit = 25;

  useEffect(() => {
    api.closedPositions(limit, offset).then(setPositions).catch(() => {});
  }, [offset]);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Closed Positions</h3>
        <div className="flex items-center gap-2 text-xs">
          <button
            onClick={() => setOffset(Math.max(0, offset - limit))}
            disabled={offset === 0}
            className="px-2 py-1 bg-surface-2 rounded text-gray-400 hover:text-gray-200 disabled:opacity-30"
          >
            Prev
          </button>
          <span className="text-gray-500">{offset + 1}-{offset + positions.length}</span>
          <button
            onClick={() => setOffset(offset + limit)}
            disabled={positions.length < limit}
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
              <th className="text-left py-2 px-2">Asset</th>
              <th className="text-left py-2 px-2">Side</th>
              <th className="text-right py-2 px-2">Size</th>
              <th className="text-right py-2 px-2">Entry</th>
              <th className="text-right py-2 px-2">Exit</th>
              <th className="text-right py-2 px-2">P&L</th>
              <th className="text-right py-2 px-2">Hold</th>
              <th className="text-left py-2 px-2">Reason</th>
              <th className="text-left py-2 px-2">Closed</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => {
              const hasPnl = p.realized_pnl != null;
              return (
                <tr key={p.position_id} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                  <td className="py-2 px-2 text-gray-300">{p.product_id}</td>
                  <td className="py-2 px-2">
                    <span className={p.side === 'BUY' || p.side === 'LONG' ? 'text-accent-green' : 'text-accent-red'}>
                      {p.side}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">{p.size.toFixed(6)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">{formatCurrency(p.entry_price)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">
                    {p.close_price != null ? formatCurrency(p.close_price) : '--'}
                  </td>
                  <td className={`py-2 px-2 text-right font-medium ${hasPnl ? pnlColor(p.realized_pnl!) : 'text-gray-500'}`}>
                    {hasPnl ? `${pnlSign(p.realized_pnl!)}${formatCurrency(p.realized_pnl!)}` : '--'}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-500">
                    {formatDuration(p.hold_duration_seconds)}
                  </td>
                  <td className="py-2 px-2">
                    {p.exit_reason ? (
                      <span className="px-1.5 py-0.5 rounded text-[10px] bg-surface-3 text-gray-400">
                        {p.exit_reason.split(':')[0]}
                      </span>
                    ) : '--'}
                  </td>
                  <td className="py-2 px-2 text-gray-500">
                    {p.closed_at ? formatTimestamp(p.closed_at) : '--'}
                  </td>
                </tr>
              );
            })}
            {positions.length === 0 && (
              <tr>
                <td colSpan={9} className="py-4 text-center text-gray-500">
                  No closed positions yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
