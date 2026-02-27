import { useEffect, useState } from 'react';
import { api } from '../api';

interface Overview {
  bankroll: number;
  bet_size: number;
  shots_left: number;
  open_count: number;
  total_bets: number;
  total_pnl: number;
  best_win: number;
}

interface Position {
  id: number;
  symbol: string;
  entry_price: number;
  current_price: number;
  peak_price: number;
  peak_gain_pct: number;
  pnl_pct: number;
  pnl_usd: number;
  bet_size_usd: number;
  entry_score: number;
  opened_at: string;
}

interface ClosedBet {
  id: number;
  symbol: string;
  entry_price: number;
  exit_price: number;
  peak_gain_pct: number;
  pnl_pct: number;
  pnl_usd: number;
  bet_size_usd: number;
  entry_score: number;
  exit_reason: string;
  opened_at: string;
  closed_at: string;
}

interface WalletEvent {
  id: number;
  event_type: string;
  amount: number;
  bankroll_after: number;
  symbol: string | null;
  detail: string | null;
  timestamp: string;
}

interface Stats {
  total_closed: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_winner_pct: number;
  avg_loser_pct: number;
  biggest_win: number;
  biggest_loss: number;
  best_peak_gain_pct: number;
  exit_reasons: Record<string, number>;
}

function pnlColor(val: number): string {
  if (val > 0) return 'text-accent-green';
  if (val < 0) return 'text-accent-red';
  return 'text-gray-400';
}

function holdHours(from: string): string {
  const ms = Date.now() - new Date(from).getTime();
  const h = ms / 3600000;
  return h < 1 ? `${Math.round(h * 60)}m` : `${h.toFixed(1)}h`;
}

function fmtPrice(p: number | null): string {
  if (p == null) return '--';
  return p < 1 ? `$${p.toFixed(6)}` : p < 100 ? `$${p.toFixed(4)}` : `$${p.toFixed(2)}`;
}

export default function BreakoutBets() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [history, setHistory] = useState<ClosedBet[]>([]);
  const [wallet, setWallet] = useState<WalletEvent[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const [ov, pos, hist, wal, st] = await Promise.all([
          api.bsOverview(),
          api.bsPositions(),
          api.bsHistory(),
          api.bsWallet(),
          api.bsStats(),
        ]);
        setOverview(ov as unknown as Overview);
        setPositions(((pos as unknown as { positions: Position[] }).positions) || []);
        setHistory(((hist as unknown as { history: ClosedBet[] }).history) || []);
        setWallet(((wal as unknown as { events: WalletEvent[] }).events) || []);
        setStats(st as unknown as Stats);
        setError('');
      } catch (e: unknown) {
        setError(String(e));
      }
    };
    load();
    const iv = setInterval(load, 10000);
    return () => clearInterval(iv);
  }, []);

  if (error) {
    return <div className="text-accent-red p-4">Error: {error}</div>;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold">Breakout Bets</h1>

      {/* ─── Top Row: Overview Cards ─── */}
      {overview && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          <Card label="Wallet" value={`$${overview.bankroll.toFixed(0)}`} />
          <Card label="Bet Size" value={`$${overview.bet_size.toFixed(0)}`} />
          <Card label="Shots Left" value={String(overview.shots_left)} />
          <Card label="Open" value={String(overview.open_count)} />
          <Card label="Total Bets" value={String(overview.total_bets)} />
          <Card label="P&L" value={`$${overview.total_pnl.toFixed(2)}`}
                className={pnlColor(overview.total_pnl)} />
          <Card label="Best Win" value={overview.best_win > 0 ? `$${overview.best_win.toFixed(2)}` : '--'} />
        </div>
      )}

      {/* ─── Open Positions ─── */}
      <Section title={`Open Positions (${positions.length})`}>
        {positions.length === 0 ? (
          <p className="text-gray-500 text-sm">No open breakout bets. Waiting for score &ge; 75 signals.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="text-left py-2">Symbol</th>
                  <th className="text-right py-2">Entry</th>
                  <th className="text-right py-2">Now</th>
                  <th className="text-right py-2">P&L%</th>
                  <th className="text-right py-2">P&L$</th>
                  <th className="text-right py-2">Peak%</th>
                  <th className="text-right py-2">Hold</th>
                  <th className="text-right py-2">Score</th>
                </tr>
              </thead>
              <tbody>
                {positions.map(p => (
                  <tr key={p.id} className="border-b border-surface-2 hover:bg-surface-2/50">
                    <td className="py-2 font-mono">{p.symbol}</td>
                    <td className="text-right font-mono">{fmtPrice(p.entry_price)}</td>
                    <td className="text-right font-mono">{fmtPrice(p.current_price)}</td>
                    <td className={`text-right font-mono ${pnlColor(p.pnl_pct)}`}>
                      {p.pnl_pct > 0 ? '+' : ''}{p.pnl_pct.toFixed(1)}%
                    </td>
                    <td className={`text-right font-mono ${pnlColor(p.pnl_usd)}`}>
                      ${p.pnl_usd.toFixed(2)}
                    </td>
                    <td className="text-right font-mono text-accent-green">
                      +{p.peak_gain_pct.toFixed(1)}%
                    </td>
                    <td className="text-right text-gray-400">{holdHours(p.opened_at)}</td>
                    <td className="text-right">{p.entry_score?.toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* ─── Stats ─── */}
      {stats && stats.total_closed > 0 && (
        <Section title="Stats">
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
            <Card label="Win Rate" value={`${stats.win_rate.toFixed(0)}%`}
                  className={stats.win_rate >= 50 ? 'text-accent-green' : 'text-accent-red'} />
            <Card label="Avg Winner" value={`+${stats.avg_winner_pct.toFixed(1)}%`}
                  className="text-accent-green" />
            <Card label="Avg Loser" value={`${stats.avg_loser_pct.toFixed(1)}%`}
                  className="text-accent-red" />
            <Card label="Biggest Win" value={`$${stats.biggest_win.toFixed(2)}`} />
            <Card label="Biggest Loss" value={`$${stats.biggest_loss.toFixed(2)}`} />
          </div>
          {Object.keys(stats.exit_reasons).length > 0 && (
            <div className="mt-3 flex gap-3 flex-wrap">
              {Object.entries(stats.exit_reasons).map(([reason, count]) => (
                <span key={reason} className="text-xs bg-surface-2 px-2 py-1 rounded">
                  {reason}: {count}
                </span>
              ))}
            </div>
          )}
        </Section>
      )}

      {/* ─── Closed Bets ─── */}
      <Section title={`Closed Bets (${history.length})`}>
        {history.length === 0 ? (
          <p className="text-gray-500 text-sm">No closed bets yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="text-left py-2">Symbol</th>
                  <th className="text-right py-2">Entry</th>
                  <th className="text-right py-2">Exit</th>
                  <th className="text-right py-2">P&L%</th>
                  <th className="text-right py-2">P&L$</th>
                  <th className="text-right py-2">Peak%</th>
                  <th className="text-left py-2">Reason</th>
                  <th className="text-right py-2">Hold</th>
                </tr>
              </thead>
              <tbody>
                {history.map(h => (
                  <tr key={h.id} className="border-b border-surface-2 hover:bg-surface-2/50">
                    <td className="py-2 font-mono">{h.symbol}</td>
                    <td className="text-right font-mono">{fmtPrice(h.entry_price)}</td>
                    <td className="text-right font-mono">{fmtPrice(h.exit_price)}</td>
                    <td className={`text-right font-mono ${pnlColor(h.pnl_pct)}`}>
                      {h.pnl_pct > 0 ? '+' : ''}{h.pnl_pct.toFixed(1)}%
                    </td>
                    <td className={`text-right font-mono ${pnlColor(h.pnl_usd)}`}>
                      ${h.pnl_usd.toFixed(2)}
                    </td>
                    <td className="text-right font-mono text-accent-green">
                      +{h.peak_gain_pct.toFixed(1)}%
                    </td>
                    <td className="text-left">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        h.exit_reason === 'trailing_stop' ? 'bg-accent-green/20 text-accent-green' :
                        h.exit_reason === 'stop_loss' ? 'bg-accent-red/20 text-accent-red' :
                        'bg-accent-yellow/20 text-accent-yellow'
                      }`}>
                        {h.exit_reason}
                      </span>
                    </td>
                    <td className="text-right text-gray-400">
                      {h.closed_at ? holdHours(h.opened_at) : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* ─── Wallet Log ─── */}
      <Section title="Wallet Log">
        {wallet.length === 0 ? (
          <p className="text-gray-500 text-sm">No wallet events yet.</p>
        ) : (
          <div className="overflow-x-auto max-h-64 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface-1">
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="text-left py-2">Time</th>
                  <th className="text-left py-2">Event</th>
                  <th className="text-left py-2">Symbol</th>
                  <th className="text-right py-2">Amount</th>
                  <th className="text-right py-2">Balance</th>
                </tr>
              </thead>
              <tbody>
                {wallet.map(w => (
                  <tr key={w.id} className="border-b border-surface-2">
                    <td className="py-1 text-gray-400 text-xs">
                      {new Date(w.timestamp).toLocaleString()}
                    </td>
                    <td className="py-1">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        w.event_type === 'seed' ? 'bg-accent-blue/20 text-accent-blue' :
                        w.event_type === 'bet_placed' ? 'bg-accent-yellow/20 text-accent-yellow' :
                        'bg-accent-green/20 text-accent-green'
                      }`}>
                        {w.event_type}
                      </span>
                    </td>
                    <td className="py-1 font-mono">{w.symbol || '--'}</td>
                    <td className={`text-right font-mono ${pnlColor(w.amount)}`}>
                      {w.amount > 0 ? '+' : ''}{w.amount.toFixed(2)}
                    </td>
                    <td className="text-right font-mono">${w.bankroll_after.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  );
}

function Card({ label, value, className = '' }: { label: string; value: string; className?: string }) {
  return (
    <div className="bg-surface-1 border border-surface-3 rounded-lg p-3">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-lg font-mono font-semibold ${className}`}>{value}</div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-surface-1 border border-surface-3 rounded-lg p-4">
      <h2 className="text-sm font-semibold text-gray-300 mb-3">{title}</h2>
      {children}
    </div>
  );
}
