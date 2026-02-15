import { useDashboard } from '../../context/DashboardContext';
import { formatCurrency, pnlColor, pnlSign } from '../../utils/formatters';

export default function OpenPositionsTable() {
  const { state } = useDashboard();
  const positions = state.positions;

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        Open Positions <span className="text-gray-600">({positions.length})</span>
      </h3>
      {positions.length === 0 ? (
        <div className="text-sm text-gray-600 py-4 text-center">No open positions</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="text-left py-2 px-2">Asset</th>
                <th className="text-left py-2 px-2">Side</th>
                <th className="text-right py-2 px-2">Size</th>
                <th className="text-right py-2 px-2">Entry</th>
                <th className="text-right py-2 px-2">Current</th>
                <th className="text-right py-2 px-2">P&L</th>
                <th className="text-right py-2 px-2">SL</th>
                <th className="text-right py-2 px-2">TP</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={p.position_id} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                  <td className="py-2 px-2 text-gray-300">{p.product_id}</td>
                  <td className="py-2 px-2">
                    <span className={p.side === 'BUY' ? 'text-accent-green' : 'text-accent-red'}>
                      {p.side}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">{p.size.toFixed(6)}</td>
                  <td className="py-2 px-2 text-right text-gray-400">{formatCurrency(p.entry_price)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">
                    {p.current_price ? formatCurrency(p.current_price) : '--'}
                  </td>
                  <td className={`py-2 px-2 text-right font-semibold ${pnlColor(p.unrealized_pnl || 0)}`}>
                    {p.unrealized_pnl != null
                      ? `${pnlSign(p.unrealized_pnl)}${formatCurrency(p.unrealized_pnl)}`
                      : '--'}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-600">
                    {p.stop_loss_price ? formatCurrency(p.stop_loss_price) : '--'}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-600">
                    {p.take_profit_price ? formatCurrency(p.take_profit_price) : '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
