import { useDashboard } from '../../context/DashboardContext';
import { useWS } from '../../context/WebSocketContext';
import { formatCurrency, formatUptime, pnlColor, pnlSign } from '../../utils/formatters';

export default function StatusStrip() {
  const { state } = useDashboard();
  const { connected } = useWS();
  const { status, pnl } = state;

  const prices = status?.latest_prices ?? {};

  return (
    <header className="h-10 bg-surface-1 border-b border-surface-3 flex items-center px-4 gap-6 text-xs font-mono shrink-0 overflow-x-auto">
      {/* Connection status */}
      <div className="flex items-center gap-1.5">
        <span className={`w-2 h-2 rounded-full ${connected ? 'bg-accent-green animate-pulse' : 'bg-accent-red'}`} />
        <span className="text-gray-400">{status?.status || 'CONNECTING'}</span>
      </div>

      {/* Uptime */}
      {status && (
        <div className="text-gray-500">
          UP {formatUptime(status.uptime_seconds)}
        </div>
      )}

      {/* Cycle count */}
      {status && (
        <div className="text-gray-500">
          Cycles: <span className="text-gray-300">{status.cycle_count}</span>
        </div>
      )}

      {/* Asset Prices — show all tracked assets */}
      {Object.entries(prices).map(([asset, snap]) => snap.price > 0 && (
        <div key={asset} className="text-gray-300 font-semibold">
          {asset.replace('-USD', '')} ${snap.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
        </div>
      ))}

      {/* Spacer */}
      <div className="flex-1" />

      {/* PnL */}
      {pnl && (
        <div className={`font-semibold ${pnlColor(pnl.realized_pnl)}`}>
          PnL: {pnlSign(pnl.realized_pnl)}{formatCurrency(pnl.realized_pnl)}
        </div>
      )}

      {/* Win Rate */}
      {pnl && pnl.total_sells > 0 && (
        <div className="text-gray-400">
          WR: <span className="text-gray-300">{(pnl.win_rate * 100).toFixed(1)}%</span>
        </div>
      )}

      {/* Trades */}
      {pnl && (
        <div className="text-gray-500">
          Trades: <span className="text-gray-300">{pnl.total_trades}</span>
        </div>
      )}

      {/* Open positions */}
      {status && (
        <div className="text-gray-500">
          Open: <span className="text-gray-300">{status.open_position_count}</span>
        </div>
      )}

      {/* WS clients */}
      <div className="text-gray-600">
        WS: {status?.ws_clients ?? 0}
      </div>
    </header>
  );
}
