import { useEffect, useState, useCallback } from 'react';
import PageShell from '../components/layout/PageShell';
import MetricCard from '../components/cards/MetricCard';
import { api } from '../api';

// ─── Types ──────────────────────────────────────────────────────────────────

interface WalletData {
  pair: string;
  capital: number;
  initial: number;
  pnl_usd: number;
  pnl_pct: number;
  position_open: boolean;
  entry_price: number;
  stop_loss_price: number;
  current_price: number;
  unrealized_pnl: number;
  total_trades: number;
  winning_trades: number;
  win_rate: number;
  max_drawdown_pct: number;
  status: string;
  last_signal: string;
}

interface Summary {
  total_capital: number;
  total_initial: number;
  total_pnl: number;
  total_return_pct: number;
  positions_open: number;
  total_pairs: number;
  active_pairs: number;
}

interface OpenTrade {
  pair: string;
  entry_price: number;
  current_price: number;
  unrealized_pnl_usd: number;
  unrealized_pnl_pct: number;
  entry_time: string;
  hours_open: number;
  signal: string;
  confidence: number;
  capital: number;
  position_size: number;
}

interface TradeRow {
  pair: string;
  action: string;
  price: number;
  capital_before: number;
  capital_after: number;
  pnl_usd: number;
  pnl_pct: number;
  exit_reason: string;
  hold_bars: number;
  signal: string;
  confidence: number;
  timestamp: string;
}

// ─── Countdown helpers ──────────────────────────────────────────────────────

function getNextPredictionTime(): Date {
  const now = new Date();
  const hour = now.getUTCHours();
  const nextBoundary = (Math.floor(hour / 4) + 1) * 4;
  const next = new Date(now);
  next.setUTCHours(nextBoundary % 24, 0, 30, 0); // +30s like the service
  if (nextBoundary >= 24) next.setUTCDate(next.getUTCDate() + 1);
  return next;
}

function formatCountdown(ms: number): string {
  if (ms <= 0) return 'Running now...';
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  return `${h}h ${m}m ${s}s`;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function fmtDollar(n: number): string {
  const abs = Math.abs(n);
  const sign = n >= 0 ? '+' : '-';
  if (abs >= 1000) return `${sign}$${abs.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
  return `${sign}$${abs.toFixed(2)}`;
}

function pnlColor(n: number): string {
  if (n > 0) return 'text-accent-green';
  if (n < 0) return 'text-accent-red';
  return 'text-gray-400';
}

function signalBadge(signal: string) {
  if (signal === 'BUY')
    return <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-accent-green/20 text-accent-green">BUY</span>;
  if (signal === 'SELL')
    return <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-accent-red/20 text-accent-red">SELL</span>;
  return <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-gray-600/40 text-gray-400">HOLD</span>;
}

function statusBadge(status: string) {
  if (status === 'live')
    return <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-accent-green/20 text-accent-green">PAPER</span>;
  if (status === 'halted')
    return <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-accent-red/20 text-accent-red">HALTED</span>;
  return <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-accent-yellow/20 text-accent-yellow">OBS</span>;
}

// ─── Component ──────────────────────────────────────────────────────────────

export default function OracleTrading() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [wallets, setWallets] = useState<WalletData[]>([]);
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [openTrades, setOpenTrades] = useState<OpenTrade[]>([]);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [countdown, setCountdown] = useState<string>('');
  const [lastRun, setLastRun] = useState<string>('');

  const fetchAll = useCallback(async () => {
    api.oracleTradingStatus().then((d: any) => {
      setSummary(d.summary || null);
      const w = Object.values(d.wallets || {}) as WalletData[];
      w.sort((a, b) => (b.pnl_usd || 0) - (a.pnl_usd || 0));
      setWallets(w);
    }).catch(() => {});

    api.oracleTradingTrades('', 30).then((d: any) => {
      setTrades(d || []);
    }).catch(() => {});

    api.oracleTradingOpenTrades().then((d: any) => {
      setOpenTrades(d || []);
    }).catch(() => {});

    api.oracleStatus().then((d: any) => {
      const allPreds: any[] = [];
      let latestTs = '';
      for (const [asset, data] of Object.entries(d.signals || {})) {
        const assetData = data as any;
        const current = assetData.current;
        if (current) {
          allPreds.push({ asset, ...current });
          const ts = current.timestamp || current.candle_time || '';
          if (ts > latestTs) latestTs = ts;
        }
      }
      allPreds.sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || ''));
      setPredictions(allPreds);
      if (latestTs) setLastRun(latestTs);
    }).catch((err) => console.error('Oracle status fetch failed:', err));
  }, []);

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 10_000);
    return () => clearInterval(id);
  }, [fetchAll]);

  // Countdown timer — ticks every second
  useEffect(() => {
    const tick = () => {
      const next = getNextPredictionTime();
      const ms = next.getTime() - Date.now();
      setCountdown(formatCountdown(ms));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <PageShell title="Oracle Trades" subtitle="Paper-exact strategy replication (Parente et al. 2023)">
      {/* Fleet Summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard
          title="Total Capital"
          value={summary ? `$${summary.total_capital.toLocaleString()}` : '--'}
          subtitle={summary ? `${summary.total_pairs} pairs` : ''}
        />
        <MetricCard
          title="Total P&L"
          value={summary ? fmtDollar(summary.total_pnl) : '--'}
          valueColor={summary ? pnlColor(summary.total_pnl) : 'text-gray-100'}
        />
        <MetricCard
          title="Return"
          value={summary ? `${summary.total_return_pct >= 0 ? '+' : ''}${summary.total_return_pct.toFixed(2)}%` : '--'}
          valueColor={summary ? pnlColor(summary.total_return_pct) : 'text-gray-100'}
        />
        <MetricCard
          title="Open Positions"
          value={summary ? String(summary.positions_open) : '--'}
        />
        <MetricCard
          title="Active Pairs"
          value={summary ? `${summary.active_pairs}/${summary.total_pairs}` : '--'}
        />
      </div>

      {/* Open Trades */}
      {openTrades.length > 0 && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-surface-3">
            <h3 className="text-sm font-medium text-gray-300">
              Open Positions <span className="text-gray-500 font-normal">({openTrades.length})</span>
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="px-3 py-2 text-left font-medium">Pair</th>
                  <th className="px-3 py-2 text-right font-medium">Invested</th>
                  <th className="px-3 py-2 text-right font-medium">Entry</th>
                  <th className="px-3 py-2 text-right font-medium">Current</th>
                  <th className="px-3 py-2 text-right font-medium">Unreal P&L</th>
                  <th className="px-3 py-2 text-right font-medium">Return</th>
                  <th className="px-3 py-2 text-right font-medium">Duration</th>
                  <th className="px-3 py-2 text-left font-medium">Signal</th>
                  <th className="px-3 py-2 text-right font-medium">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {openTrades.map(t => {
                  const dur = t.hours_open;
                  const durStr = dur >= 24
                    ? `${Math.floor(dur / 24)}d ${Math.floor(dur % 24)}h`
                    : dur >= 1
                      ? `${Math.floor(dur)}h ${Math.floor((dur % 1) * 60)}m`
                      : `${Math.floor(dur * 60)}m`;
                  return (
                    <tr key={t.pair} className="border-b border-surface-3/50 hover:bg-surface-2/30">
                      <td className="px-3 py-2 font-mono font-medium text-gray-200">
                        {t.pair.replace('USDT', '')}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-gray-300">
                        ${t.position_size.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-gray-300">
                        ${t.entry_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-gray-300">
                        ${t.current_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${pnlColor(t.unrealized_pnl_usd)}`}>
                        {fmtDollar(t.unrealized_pnl_usd)}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${pnlColor(t.unrealized_pnl_pct)}`}>
                        {t.unrealized_pnl_pct >= 0 ? '+' : ''}{t.unrealized_pnl_pct.toFixed(2)}%
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-gray-400">
                        {durStr}
                      </td>
                      <td className="px-3 py-2">{signalBadge(t.signal)}</td>
                      <td className="px-3 py-2 text-right font-mono text-gray-400">
                        {t.confidence ? `${(t.confidence * 100).toFixed(0)}%` : '--'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Countdown Timer + Last Run Info */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <span className="text-xs text-gray-500 mr-2">Next Prediction</span>
            <span className="text-sm font-mono font-medium text-gray-100">{countdown}</span>
          </div>
          <div>
            <span className="text-xs text-gray-500 mr-2">Last Run</span>
            <span className="text-sm font-mono text-gray-300">
              {lastRun ? lastRun.replace('T', ' ').slice(0, 19) + ' UTC' : '--'}
            </span>
          </div>
        </div>
        <div className="text-xs text-gray-600">4H boundaries: 00:00 / 04:00 / 08:00 / 12:00 / 16:00 / 20:00 UTC</div>
      </div>

      {/* Prediction Log */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-surface-3">
          <h3 className="text-sm font-medium text-gray-300">Prediction Log</h3>
        </div>
        <div className="divide-y divide-surface-3/50">
          {predictions.length === 0 && (
            <div className="px-4 py-6 text-center text-gray-500 text-xs">
              Waiting for oracle predictions...
            </div>
          )}
          {predictions.map((pred, idx) => {
            const signal = (pred.signal || pred.action || 'HOLD').toUpperCase();
            const confidence = pred.confidence != null ? (pred.confidence * 100).toFixed(1) : '--';
            const closePrice = pred.close_price || pred.candle_close || pred.price;
            const candleTime = pred.candle_time || pred.timestamp || '';
            const modelVotes: any[] = Array.isArray(pred.model_votes) ? pred.model_votes : [];

            return (
              <div key={idx} className="px-4 py-3">
                {/* Asset header row */}
                <div className="flex items-center gap-3 mb-2">
                  <span className="font-mono font-medium text-gray-200 text-sm">
                    {(pred.asset || '').replace('USDT', '')}
                  </span>
                  {signalBadge(signal)}
                  <span className="text-xs text-gray-400">
                    Confidence: <span className="font-mono text-gray-200">{confidence}%</span>
                  </span>
                  {closePrice != null && (
                    <span className="text-xs text-gray-400">
                      Close: <span className="font-mono text-gray-300">
                        ${Number(closePrice).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </span>
                    </span>
                  )}
                  {candleTime && (
                    <span className="text-xs text-gray-500 font-mono ml-auto">
                      {candleTime.replace('T', ' ').slice(0, 19)}
                    </span>
                  )}
                </div>

                {/* Model vote breakdown */}
                {modelVotes.length > 0 && (
                  <div className="space-y-1 ml-1">
                    {modelVotes.map((vote: any, vi: number) => {
                      const shortName = (vote.model || '')
                        .replace('model_final_', '')
                        .replace('.h5', '')
                        .replace(/_/g, '\u00D7');
                      const weight = vote.weight != null ? (vote.weight * 100).toFixed(0) : '?';
                      const mSignal = (vote.signal || 'HOLD').toUpperCase();
                      const probs = vote.probs || [0, 0, 0];
                      const buyPct = (probs[0] || 0) * 100;
                      const holdPct = (probs[1] || 0) * 100;
                      const sellPct = (probs[2] || 0) * 100;

                      return (
                        <div key={vi} className="flex items-center gap-2 text-xs">
                          <span className="font-mono text-gray-400 w-12 text-right">{shortName}</span>
                          <span className="text-gray-500 w-10 text-right">({weight}%)</span>
                          <span className={`w-8 text-center font-medium ${
                            mSignal === 'BUY' ? 'text-accent-green' :
                            mSignal === 'SELL' ? 'text-accent-red' :
                            'text-gray-400'
                          }`}>
                            {mSignal}
                          </span>
                          <div className="flex h-1.5 rounded-full overflow-hidden bg-surface-3 w-24">
                            <div className="bg-accent-green" style={{ width: `${buyPct}%` }} />
                            <div className="bg-gray-500" style={{ width: `${holdPct}%` }} />
                            <div className="bg-accent-red" style={{ width: `${sellPct}%` }} />
                          </div>
                          <span className="font-mono text-gray-500">
                            B:{buyPct.toFixed(1)}% H:{holdPct.toFixed(1)}% S:{sellPct.toFixed(1)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Wallet Table */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-surface-3">
          <h3 className="text-sm font-medium text-gray-300">Oracle Wallets</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="px-3 py-2 text-left font-medium">Pair</th>
                <th className="px-3 py-2 text-left font-medium">Signal</th>
                <th className="px-3 py-2 text-right font-medium">Capital</th>
                <th className="px-3 py-2 text-right font-medium">P&L</th>
                <th className="px-3 py-2 text-right font-medium">Return</th>
                <th className="px-3 py-2 text-right font-medium">Trades</th>
                <th className="px-3 py-2 text-right font-medium">Win Rate</th>
                <th className="px-3 py-2 text-left font-medium">Position</th>
                <th className="px-3 py-2 text-right font-medium">Max DD</th>
                <th className="px-3 py-2 text-left font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {wallets.length === 0 && (
                <tr><td colSpan={10} className="px-3 py-6 text-center text-gray-500">
                  Waiting for first cycle...
                </td></tr>
              )}
              {wallets.map(w => (
                <tr key={w.pair} className="border-b border-surface-3/50 hover:bg-surface-2/30">
                  <td className="px-3 py-2 font-mono font-medium text-gray-200">
                    {w.pair.replace('USDT', '')}
                  </td>
                  <td className="px-3 py-2">{signalBadge(w.last_signal)}</td>
                  <td className="px-3 py-2 text-right font-mono text-gray-300">
                    ${w.capital.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono ${pnlColor(w.pnl_usd)}`}>
                    {fmtDollar(w.pnl_usd)}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono ${pnlColor(w.pnl_pct)}`}>
                    {w.pnl_pct >= 0 ? '+' : ''}{w.pnl_pct.toFixed(2)}%
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-400">
                    {w.total_trades}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-400">
                    {w.win_rate.toFixed(0)}%
                  </td>
                  <td className="px-3 py-2 text-gray-400">
                    {w.position_open ? (
                      <span className="text-accent-blue">
                        LONG @ ${w.entry_price.toLocaleString()}
                        {w.unrealized_pnl !== 0 && (
                          <span className={`ml-1 ${pnlColor(w.unrealized_pnl)}`}>
                            ({w.unrealized_pnl >= 0 ? '+' : ''}{w.unrealized_pnl}%)
                          </span>
                        )}
                      </span>
                    ) : (
                      <span className="text-gray-600">flat</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-400">
                    {w.max_drawdown_pct.toFixed(1)}%
                  </td>
                  <td className="px-3 py-2">{statusBadge(w.status)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent Trades */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-surface-3">
          <h3 className="text-sm font-medium text-gray-300">Recent Trades</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="px-3 py-2 text-left font-medium">Time</th>
                <th className="px-3 py-2 text-left font-medium">Pair</th>
                <th className="px-3 py-2 text-left font-medium">Action</th>
                <th className="px-3 py-2 text-right font-medium">Price</th>
                <th className="px-3 py-2 text-right font-medium">P&L</th>
                <th className="px-3 py-2 text-right font-medium">Return</th>
                <th className="px-3 py-2 text-left font-medium">Reason</th>
                <th className="px-3 py-2 text-right font-medium">Held</th>
                <th className="px-3 py-2 text-right font-medium">Capital</th>
              </tr>
            </thead>
            <tbody>
              {trades.length === 0 && (
                <tr><td colSpan={9} className="px-3 py-6 text-center text-gray-500">
                  No trades yet — waiting for BUY signals
                </td></tr>
              )}
              {trades.map((t, i) => (
                <tr key={i} className="border-b border-surface-3/50 hover:bg-surface-2/30">
                  <td className="px-3 py-2 text-gray-500 font-mono">
                    {t.timestamp?.slice(5, 16)}
                  </td>
                  <td className="px-3 py-2 font-mono font-medium text-gray-300">
                    {t.pair?.replace('USDT', '')}
                  </td>
                  <td className="px-3 py-2">
                    {t.action === 'OPEN' ? (
                      <span className="text-accent-blue font-medium">OPEN</span>
                    ) : (
                      <span className="text-gray-300 font-medium">CLOSE</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-300">
                    ${t.price?.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono ${t.action === 'CLOSE' ? pnlColor(t.pnl_usd || 0) : 'text-gray-600'}`}>
                    {t.action === 'CLOSE' ? fmtDollar(t.pnl_usd || 0) : '--'}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono ${t.action === 'CLOSE' ? pnlColor(t.pnl_pct || 0) : 'text-gray-600'}`}>
                    {t.action === 'CLOSE' ? `${(t.pnl_pct || 0) >= 0 ? '+' : ''}${((t.pnl_pct || 0) * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="px-3 py-2 text-gray-400">
                    {t.exit_reason || '--'}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-400">
                    {t.hold_bars != null && t.action === 'CLOSE' ? `${t.hold_bars} bars` : '--'}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-300">
                    {t.action === 'CLOSE' && t.capital_after
                      ? `$${t.capital_after.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                      : t.capital_before
                        ? `$${t.capital_before.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                        : '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </PageShell>
  );
}
