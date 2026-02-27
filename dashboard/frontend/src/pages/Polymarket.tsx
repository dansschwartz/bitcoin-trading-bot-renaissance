import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import MetricCard from '../components/cards/MetricCard';
import { api } from '../api';
import { formatCurrency, formatTimestamp, pnlColor } from '../utils/formatters';

/* ── Types ─────────────────────────────────────────────────────── */

interface Overview {
  bankroll: number;
  initial_bankroll: number;
  open_count: number;
  open_exposure: number;
  today_pnl: number;
  today_bets: number;
  today_wins: number;
  total_pnl: number;
  total_bets: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_wagered: number;
}

interface Position {
  position_id: string;
  slug: string;
  asset: string;
  direction: string;
  entry_price: number;
  shares: number;
  bet_amount: number;
  status: string;
  pnl: number | null;
  exit_price: number | null;
  opened_at: string;
  closed_at: string | null;
  unrealized_pnl: number | null;
  notes: string | null;
}

interface ClosedBet {
  position_id: string;
  slug: string;
  asset: string;
  direction: string;
  entry_price: number;
  exit_price: number | null;
  bet_amount: number;
  pnl: number | null;
  status: string;
  opened_at: string;
  closed_at: string | null;
}

interface Instrument {
  key: string;
  asset: string;
  ml_pair: string;
  enabled: boolean;
  lead_asset: string | null;
}

interface Stats {
  total_bets: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_return: number;
  best_trade: number;
  worst_trade: number;
  per_asset: { asset: string; bets: number; wins: number; pnl: number; avg_pnl: number }[];
  recent_activity: ActivityEntry[];
}

interface ActivityEntry {
  timestamp: string;
  asset: string;
  direction: string;
  action: string;
  ml_confidence: number;
  token_cost: number;
  amount_usd: number;
  notes: string | null;
}

/* ── Main Page ─────────────────────────────────────────────────── */

export default function Polymarket() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [history, setHistory] = useState<ClosedBet[]>([]);
  const [instruments, setInstruments] = useState<Instrument[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    const load = () => {
      api.pmOverview().then((d) => setOverview(d as unknown as Overview)).catch(() => {});
      api.pmPositions('open').then((d) =>
        setPositions(((d as Record<string, unknown>).positions ?? []) as Position[])
      ).catch(() => {});
      api.pmHistory(20).then((d) =>
        setHistory(((d as Record<string, unknown>).bets ?? []) as ClosedBet[])
      ).catch(() => {});
      api.pmInstruments().then((d) =>
        setInstruments(((d as Record<string, unknown>).instruments ?? []) as Instrument[])
      ).catch(() => {});
      api.pmStats().then((d) => setStats(d as unknown as Stats)).catch(() => {});
    };
    load();
    const id = setInterval(load, 15_000);
    return () => clearInterval(id);
  }, []);

  const openCount = positions.length;
  const bankrollChange = overview ? overview.bankroll - overview.initial_bankroll : 0;

  return (
    <PageShell
      title="Polymarket"
      subtitle="Strategy A v2 — Confidence-gated entry"
      actions={
        <span className="px-2 py-1 rounded text-xs font-medium bg-accent-yellow/20 text-accent-yellow">
          PAPER
        </span>
      }
    >
      {/* 1. Bankroll Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <MetricCard
          title="Bankroll"
          value={overview ? formatCurrency(overview.bankroll) : '$500'}
          subtitle={overview ? `${bankrollChange >= 0 ? '+' : ''}${formatCurrency(bankrollChange)}` : undefined}
          valueColor={bankrollChange >= 0 ? 'text-accent-green' : 'text-accent-red'}
        />
        <MetricCard
          title="Today P&L"
          value={overview ? `$${overview.today_pnl.toFixed(2)}` : '$0'}
          subtitle={overview && overview.today_bets > 0 ? `${overview.today_bets} bets` : undefined}
          valueColor={pnlColor(overview?.today_pnl ?? 0)}
        />
        <MetricCard
          title="Total P&L"
          value={overview ? `$${overview.total_pnl.toFixed(2)}` : '$0'}
          valueColor={pnlColor(overview?.total_pnl ?? 0)}
        />
        <MetricCard
          title="Win Rate"
          value={overview && overview.total_bets > 0
            ? `${overview.win_rate}%`
            : '--'}
          subtitle={overview && overview.total_bets > 0
            ? `${overview.wins}W / ${overview.losses}L`
            : undefined}
          valueColor={
            (overview?.win_rate ?? 0) >= 55 ? 'text-accent-green' :
            (overview?.win_rate ?? 0) >= 45 ? 'text-gray-300' :
            'text-accent-red'
          }
        />
        <MetricCard
          title="Open Bets"
          value={`${openCount}`}
          subtitle={overview ? `$${overview.open_exposure.toFixed(0)} exposed` : undefined}
        />
        <MetricCard
          title="Total Wagered"
          value={overview ? formatCurrency(overview.total_wagered) : '$0'}
        />
      </div>

      {/* 2. Open Positions */}
      <OpenPositions positions={positions} />

      {/* 3. Recent Bets */}
      <RecentBets bets={history} />

      {/* 4. Instrument Cards */}
      <InstrumentCards instruments={instruments} stats={stats} />

      {/* 5. P&L by Asset */}
      {stats && stats.per_asset.length > 0 && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">P&L by Asset</h3>
          <div className="flex flex-wrap gap-3">
            {stats.per_asset.map((a) => (
              <div key={a.asset} className="bg-surface-2 rounded-lg p-3 min-w-[130px]">
                <div className="text-xs text-gray-500">{a.asset}</div>
                <div className={`text-lg font-mono ${pnlColor(a.pnl)}`}>
                  ${a.pnl.toFixed(2)}
                </div>
                <div className="text-[10px] text-gray-500">
                  {a.wins}W / {a.bets - a.wins}L | avg ${a.avg_pnl.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
          {stats.total_bets > 0 && (
            <div className="mt-3 grid grid-cols-3 gap-3 text-xs font-mono">
              <div className="bg-surface-2 rounded-lg p-2 text-center">
                <div className="text-gray-500">Avg Return</div>
                <div className={pnlColor(stats.avg_return)}>${stats.avg_return.toFixed(2)}</div>
              </div>
              <div className="bg-surface-2 rounded-lg p-2 text-center">
                <div className="text-gray-500">Best</div>
                <div className="text-accent-green">${stats.best_trade.toFixed(2)}</div>
              </div>
              <div className="bg-surface-2 rounded-lg p-2 text-center">
                <div className="text-gray-500">Worst</div>
                <div className="text-accent-red">${stats.worst_trade.toFixed(2)}</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 6. Activity Log */}
      <ActivityLog activity={stats?.recent_activity ?? []} />
    </PageShell>
  );
}

/* ── Open Positions ────────────────────────────────────────────── */

function OpenPositions({ positions }: { positions: Position[] }) {
  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        Open Positions ({positions.length})
      </h3>
      {positions.length === 0 ? (
        <div className="text-center text-gray-500 text-xs py-4">
          No open bets. Waiting for ML confidence &ge; 90%.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="text-left py-2 px-2">Asset</th>
                <th className="text-left py-2 px-2">Dir</th>
                <th className="text-right py-2 px-2">Entry</th>
                <th className="text-right py-2 px-2">Shares</th>
                <th className="text-right py-2 px-2">Bet</th>
                <th className="text-right py-2 px-2">Unreal. P&L</th>
                <th className="text-left py-2 px-2">Age</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={p.position_id} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                  <td className="py-2 px-2 text-gray-200 font-medium">{p.asset}</td>
                  <td className="py-2 px-2">
                    <span className={p.direction === 'UP' ? 'text-accent-green' : 'text-accent-red'}>
                      {p.direction}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">${p.entry_price.toFixed(3)}</td>
                  <td className="py-2 px-2 text-right text-gray-400">{p.shares.toFixed(1)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">${p.bet_amount.toFixed(2)}</td>
                  <td className={`py-2 px-2 text-right ${p.unrealized_pnl != null ? pnlColor(p.unrealized_pnl) : 'text-gray-500'}`}>
                    {p.unrealized_pnl != null ? `$${p.unrealized_pnl.toFixed(2)}` : '--'}
                  </td>
                  <td className="py-2 px-2 text-gray-500">{formatTimestamp(p.opened_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ── Recent Bets ───────────────────────────────────────────────── */

function RecentBets({ bets }: { bets: ClosedBet[] }) {
  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        Recent Bets ({bets.length})
      </h3>
      {bets.length === 0 ? (
        <div className="text-center text-gray-500 text-xs py-4">
          No resolved bets yet.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="text-left py-2 px-2">Asset</th>
                <th className="text-left py-2 px-2">Dir</th>
                <th className="text-right py-2 px-2">Entry</th>
                <th className="text-right py-2 px-2">Exit</th>
                <th className="text-right py-2 px-2">Bet</th>
                <th className="text-right py-2 px-2">P&L</th>
                <th className="text-left py-2 px-2">Result</th>
                <th className="text-left py-2 px-2">Closed</th>
              </tr>
            </thead>
            <tbody>
              {bets.map((b) => (
                <tr key={b.position_id} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                  <td className="py-2 px-2 text-gray-200 font-medium">{b.asset}</td>
                  <td className="py-2 px-2">
                    <span className={b.direction === 'UP' ? 'text-accent-green' : 'text-accent-red'}>
                      {b.direction}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">${b.entry_price.toFixed(3)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">
                    {b.exit_price != null ? `$${b.exit_price.toFixed(3)}` : '--'}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">${b.bet_amount.toFixed(2)}</td>
                  <td className={`py-2 px-2 text-right font-medium ${pnlColor(b.pnl ?? 0)}`}>
                    {b.pnl != null ? `$${b.pnl.toFixed(2)}` : '--'}
                  </td>
                  <td className="py-2 px-2">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                      b.status === 'won' ? 'bg-accent-green/20 text-accent-green' :
                      b.status === 'lost' ? 'bg-accent-red/20 text-accent-red' :
                      b.status === 'sold' ? 'bg-accent-blue/20 text-accent-blue' :
                      'bg-surface-3 text-gray-400'
                    }`}>
                      {b.status.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-gray-500">
                    {b.closed_at ? formatTimestamp(b.closed_at) : '--'}
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

/* ── Instrument Cards ──────────────────────────────────────────── */

function InstrumentCards({ instruments, stats }: { instruments: Instrument[]; stats: Stats | null }) {
  if (instruments.length === 0) return null;

  const assetStats = new Map(
    (stats?.per_asset ?? []).map((a) => [a.asset, a])
  );

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Instruments</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {instruments.map((inst) => {
          const s = assetStats.get(inst.asset);
          return (
            <div
              key={inst.key}
              className={`bg-surface-2 rounded-lg p-3 border ${
                inst.enabled ? 'border-accent-green/30' : 'border-surface-3 opacity-50'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-200">{inst.asset}</span>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                  inst.enabled ? 'bg-accent-green/20 text-accent-green' : 'bg-gray-600 text-gray-400'
                }`}>
                  {inst.enabled ? 'ON' : 'OFF'}
                </span>
              </div>
              <div className="space-y-1 text-xs font-mono text-gray-400">
                <div className="flex justify-between">
                  <span>ML Pair</span><span>{inst.ml_pair}</span>
                </div>
                {inst.lead_asset && (
                  <div className="flex justify-between">
                    <span>Lead</span><span>{inst.lead_asset}</span>
                  </div>
                )}
                {s && (
                  <>
                    <div className="flex justify-between">
                      <span>Bets</span><span>{s.bets}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Win Rate</span>
                      <span className={pnlColor(s.wins / Math.max(s.bets, 1) - 0.5)}>
                        {s.bets > 0 ? `${(s.wins / s.bets * 100).toFixed(0)}%` : '--'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>P&L</span>
                      <span className={pnlColor(s.pnl)}>${s.pnl.toFixed(2)}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Activity Log ──────────────────────────────────────────────── */

function ActivityLog({ activity }: { activity: ActivityEntry[] }) {
  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        Activity Log ({activity.length})
      </h3>
      {activity.length === 0 ? (
        <div className="text-center text-gray-500 text-xs py-4">
          No activity yet. Strategy A hasn't run.
        </div>
      ) : (
        <div className="space-y-1">
          {activity.map((a, i) => (
            <div key={i} className="flex items-center gap-2 text-xs font-mono py-1 border-b border-surface-3/30">
              <span className="text-gray-500 w-16 shrink-0">{formatTimestamp(a.timestamp)}</span>
              <span className="text-gray-200 font-medium w-10">{a.asset}</span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] w-12 text-center ${
                a.action === 'BUY' ? 'bg-accent-green/20 text-accent-green' :
                a.action === 'SELL' ? 'bg-accent-red/20 text-accent-red' :
                a.action === 'ADD' ? 'bg-accent-blue/20 text-accent-blue' :
                'bg-surface-3 text-gray-400'
              }`}>
                {a.action}
              </span>
              <span className={a.direction === 'UP' ? 'text-accent-green' : 'text-accent-red'}>
                {a.direction}
              </span>
              <span className="text-gray-400">
                conf={a.ml_confidence.toFixed(0)}%
              </span>
              {a.token_cost > 0 && (
                <span className="text-gray-500">
                  token=${a.token_cost.toFixed(2)}
                </span>
              )}
              {a.amount_usd > 0 && (
                <span className="text-gray-300">${a.amount_usd.toFixed(2)}</span>
              )}
              {a.notes && (
                <span className="text-gray-600 truncate max-w-[200px]" title={a.notes}>
                  {a.notes}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
