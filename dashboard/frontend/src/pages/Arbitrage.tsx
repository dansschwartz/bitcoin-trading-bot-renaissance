import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import MetricCard from '../components/cards/MetricCard';
import { api } from '../api';
import type { ArbStatus, ArbTrade, ArbSummary, ArbWallet } from '../types';
import { formatCurrency, formatTimestamp, pnlColor, pnlSign, formatUptime } from '../utils/formatters';

const STRATEGY_COLORS: Record<string, string> = {
  cross_exchange: 'bg-accent-blue/20 text-accent-blue',
  triangular: 'bg-purple-400/20 text-purple-400',
  funding_rate: 'bg-accent-green/20 text-accent-green',
};

const STATUS_COLORS: Record<string, string> = {
  filled: 'bg-accent-green/20 text-accent-green',
  partial: 'bg-accent-yellow/20 text-accent-yellow',
  insufficient_balance: 'bg-gray-600/40 text-gray-400',
  failed: 'bg-accent-red/20 text-accent-red',
  one_sided_fill: 'bg-accent-red/20 text-accent-red',
};

export default function Arbitrage() {
  const [status, setStatus] = useState<ArbStatus | null>(null);
  const [trades, setTrades] = useState<ArbTrade[]>([]);
  const [summary, setSummary] = useState<ArbSummary | null>(null);
  const [wallet, setWallet] = useState<ArbWallet | null>(null);
  const [showSignals, setShowSignals] = useState(false);

  useEffect(() => {
    const load = () => {
      api.arbStatus().then(setStatus).catch(() => {});
      api.arbTrades(50).then(setTrades).catch(() => {});
      api.arbSummary().then(setSummary).catch(() => {});
      api.arbWallet().then(setWallet).catch(() => {});
    };
    load();
    const id = setInterval(load, 10_000);
    return () => clearInterval(id);
  }, []);

  const isRunning = status?.running ?? false;
  const uptime = status?.uptime_seconds ?? 0;
  const booksConnected = (status?.book_status as Record<string, unknown>)?.tradeable_pairs ?? 0;

  return (
    <PageShell
      title="Arbitrage Engine"
      subtitle="Cross-exchange, triangular, and funding rate arbitrage"
      actions={
        <div className="flex items-center gap-2">
          <span
            className={`px-2 py-1 rounded text-xs font-medium ${
              isRunning ? 'bg-accent-green/20 text-accent-green' : 'bg-gray-600/40 text-gray-400'
            }`}
          >
            {isRunning ? 'RUNNING' : 'STOPPED'}
          </span>
          {isRunning && (
            <span className="text-xs text-gray-500">
              Uptime: {formatUptime(uptime)} | Books: {String(booksConnected)}
            </span>
          )}
        </div>
      }
    >
      {/* Wallet Section */}
      {wallet && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Arbitrage Wallet</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <div className="text-xs text-gray-500">Initial Balance</div>
              <div className="text-lg font-mono text-gray-200">{formatCurrency(wallet.initial_balance)}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Current Balance</div>
              <div className={`text-lg font-mono ${pnlColor(wallet.total_realized_pnl)}`}>
                {formatCurrency(wallet.current_balance)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Realized P&L</div>
              <div className={`text-lg font-mono ${pnlColor(wallet.total_realized_pnl)}`}>
                {pnlSign(wallet.total_realized_pnl)}{formatCurrency(wallet.total_realized_pnl)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Return</div>
              <div className={`text-lg font-mono ${pnlColor(wallet.return_pct)}`}>
                {pnlSign(wallet.return_pct)}{wallet.return_pct.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          title="Total Trades"
          value={summary?.total_trades ?? 0}
          subtitle={`${summary?.filled_trades ?? 0} filled`}
        />
        <MetricCard
          title="Net P&L"
          value={formatCurrency(summary?.total_profit_usd ?? 0)}
          valueColor={pnlColor(summary?.total_profit_usd ?? 0)}
        />
        <MetricCard
          title="Win Rate"
          value={`${((summary?.win_rate ?? 0) * 100).toFixed(1)}%`}
          subtitle={`${summary?.wins ?? 0}W / ${summary?.losses ?? 0}L`}
          valueColor={(summary?.win_rate ?? 0) >= 0.5 ? 'text-accent-green' : 'text-accent-red'}
        />
        <MetricCard
          title="Signals"
          value={summary?.signals_total ?? 0}
          subtitle={`${summary?.signals_approved ?? 0} approved`}
        />
        <MetricCard
          title="Daily P&L"
          value={formatCurrency(summary?.daily_pnl_usd ?? 0)}
          valueColor={pnlColor(summary?.daily_pnl_usd ?? 0)}
        />
        <MetricCard
          title="Filled Rate"
          value={
            summary && summary.total_trades > 0
              ? `${((summary.filled_trades / summary.total_trades) * 100).toFixed(1)}%`
              : '0%'
          }
        />
      </div>

      {/* Strategy Breakdown */}
      {summary?.by_strategy && summary.by_strategy.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {summary.by_strategy.map((s) => (
            <div key={s.strategy} className="bg-surface-1 border border-surface-3 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${STRATEGY_COLORS[s.strategy] || 'bg-surface-3 text-gray-400'}`}>
                  {s.strategy.replace(/_/g, ' ')}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <div className="text-gray-500">Trades</div>
                  <div className="text-lg font-mono text-gray-200">{s.trades}</div>
                </div>
                <div>
                  <div className="text-gray-500">Fills</div>
                  <div className="text-lg font-mono text-gray-200">{s.fills}</div>
                </div>
                <div>
                  <div className="text-gray-500">Profit</div>
                  <div className={`text-lg font-mono ${pnlColor(s.profit_usd)}`}>
                    {formatCurrency(s.profit_usd)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Recent Trades Table */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Recent Trades</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="text-left py-2 px-2">Time</th>
                <th className="text-left py-2 px-2">Strategy</th>
                <th className="text-left py-2 px-2">Symbol</th>
                <th className="text-left py-2 px-2">Buy/Sell</th>
                <th className="text-right py-2 px-2">Spread (bps)</th>
                <th className="text-left py-2 px-2">Status</th>
                <th className="text-right py-2 px-2">P&L</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => (
                <tr key={t.id} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                  <td className="py-2 px-2 text-gray-500">{formatTimestamp(t.timestamp)}</td>
                  <td className="py-2 px-2">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${STRATEGY_COLORS[t.strategy] || 'bg-surface-3 text-gray-400'}`}>
                      {t.strategy}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-gray-300">{t.symbol}</td>
                  <td className="py-2 px-2 text-gray-400">
                    {t.buy_exchange} / {t.sell_exchange}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">
                    {t.net_spread_bps?.toFixed(1) ?? '--'}
                  </td>
                  <td className="py-2 px-2">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${STATUS_COLORS[t.status] || 'bg-surface-3 text-gray-400'}`}>
                      {t.status}
                    </span>
                  </td>
                  <td className={`py-2 px-2 text-right ${pnlColor(t.actual_profit_usd)}`}>
                    {t.actual_profit_usd != null
                      ? `${pnlSign(t.actual_profit_usd)}${formatCurrency(t.actual_profit_usd)}`
                      : '--'}
                  </td>
                </tr>
              ))}
              {trades.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-4 text-center text-gray-500">
                    No arbitrage trades recorded
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Signals (collapsible) */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <button
          onClick={() => setShowSignals(!showSignals)}
          className="flex items-center gap-2 text-sm font-medium text-gray-300 hover:text-gray-100"
        >
          <svg className={`w-4 h-4 transition-transform ${showSignals ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          Recent Signals
        </button>
        {showSignals && <SignalsTable />}
      </div>
    </PageShell>
  );
}

function SignalsTable() {
  const [signals, setSignals] = useState<Array<Record<string, unknown>>>([]);

  useEffect(() => {
    api.arbSignals(100).then((d) => setSignals(d as unknown as Array<Record<string, unknown>>)).catch(() => {});
  }, []);

  if (signals.length === 0) {
    return <p className="text-xs text-gray-500 mt-2">No signals recorded yet.</p>;
  }

  return (
    <div className="overflow-x-auto mt-3">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-gray-500 border-b border-surface-3">
            <th className="text-left py-2 px-2">Time</th>
            <th className="text-left py-2 px-2">Strategy</th>
            <th className="text-left py-2 px-2">Symbol</th>
            <th className="text-right py-2 px-2">Gross (bps)</th>
            <th className="text-right py-2 px-2">Net (bps)</th>
            <th className="text-center py-2 px-2">Approved</th>
            <th className="text-center py-2 px-2">Executed</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s, i) => (
            <tr key={i} className="border-b border-surface-3/50">
              <td className="py-2 px-2 text-gray-500">{formatTimestamp(String(s.timestamp ?? ''))}</td>
              <td className="py-2 px-2">
                <span className={`px-1.5 py-0.5 rounded text-[10px] ${STRATEGY_COLORS[String(s.strategy ?? '')] || 'bg-surface-3 text-gray-400'}`}>
                  {String(s.strategy ?? '')}
                </span>
              </td>
              <td className="py-2 px-2 text-gray-300">{String(s.symbol ?? '')}</td>
              <td className="py-2 px-2 text-right text-gray-300">
                {typeof s.gross_spread_bps === 'number' ? s.gross_spread_bps.toFixed(1) : '--'}
              </td>
              <td className="py-2 px-2 text-right text-gray-300">
                {typeof s.net_spread_bps === 'number' ? s.net_spread_bps.toFixed(1) : '--'}
              </td>
              <td className="py-2 px-2 text-center">
                <span className={s.approved ? 'text-accent-green' : 'text-gray-500'}>
                  {s.approved ? 'Y' : 'N'}
                </span>
              </td>
              <td className="py-2 px-2 text-center">
                <span className={s.executed ? 'text-accent-green' : 'text-gray-500'}>
                  {s.executed ? 'Y' : 'N'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
