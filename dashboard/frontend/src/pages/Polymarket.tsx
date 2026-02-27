import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import MetricCard from '../components/cards/MetricCard';
import { api } from '../api';
import { formatCurrency, formatTimestamp, pnlColor } from '../utils/formatters';

/* -- Types ---------------------------------------------------------- */

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

interface Bet {
  id: number;
  slug: string;
  asset: string;
  entry_side: string;
  entry_token_cost: number;
  entry_amount: number;
  entry_tokens: number;
  entry_confidence: number;
  adds_count: number;
  total_invested: number;
  total_tokens: number;
  avg_cost: number;
  status: string;
  exit_price: number | null;
  exit_reason: string | null;
  exit_at: string | null;
  pnl: number | null;
  return_pct: number | null;
  regime: string | null;
  opened_at: string;
  question: string | null;
}

interface LiveMarket {
  asset: string;
  slug: string;
  question: string;
  yes_price: number;
  no_price: number;
  minutes_left: number | null;
  deadline: string;
  resolved: boolean;
  volume_24h: number;
  our_bet: { entry_side: string; total_invested: number; avg_cost: number; status: string } | null;
}

interface CalibrationBucket {
  label: string;
  total: number;
  wins: number;
  accuracy: number;
}

interface SkipEntry {
  timestamp: string;
  asset: string;
  slug: string | null;
  reason: string;
  ml_confidence: number;
  token_cost: number;
  ml_direction: string;
  minutes_left: number;
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
}

/* -- Main Page ------------------------------------------------------ */

export default function Polymarket() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [openBets, setOpenBets] = useState<Bet[]>([]);
  const [history, setHistory] = useState<Bet[]>([]);
  const [liveMarkets, setLiveMarkets] = useState<LiveMarket[]>([]);
  const [calibration, setCalibration] = useState<CalibrationBucket[]>([]);
  const [skips, setSkips] = useState<SkipEntry[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    const load = () => {
      api.pmOverview().then((d) => setOverview(d as unknown as Overview)).catch(() => {});
      api.pmPositions('open').then((d) =>
        setOpenBets(((d as Record<string, unknown>).positions ?? []) as Bet[])
      ).catch(() => {});
      api.pmHistory(20).then((d) =>
        setHistory(((d as Record<string, unknown>).bets ?? []) as Bet[])
      ).catch(() => {});
      api.pmLiveMarkets().then((d) =>
        setLiveMarkets(((d as Record<string, unknown>).markets ?? []) as LiveMarket[])
      ).catch(() => {});
      api.pmCalibration().then((d) =>
        setCalibration(((d as Record<string, unknown>).buckets ?? []) as CalibrationBucket[])
      ).catch(() => {});
      api.pmSkipLog(20).then((d) =>
        setSkips(((d as Record<string, unknown>).skips ?? []) as SkipEntry[])
      ).catch(() => {});
      api.pmStats().then((d) => setStats(d as unknown as Stats)).catch(() => {});
    };
    load();
    const id = setInterval(load, 15_000);
    return () => clearInterval(id);
  }, []);

  const bankrollChange = overview ? overview.bankroll - overview.initial_bankroll : 0;

  return (
    <PageShell
      title="Polymarket"
      subtitle="Strategy A v3 — Confidence-gated entry"
      actions={
        <span className="px-2 py-1 rounded text-xs font-medium bg-accent-yellow/20 text-accent-yellow">
          PAPER
        </span>
      }
    >
      {/* 1. Bankroll Bar */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
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
          value={overview && overview.total_bets > 0 ? `${overview.win_rate}%` : '--'}
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
          value={`${openBets.length}`}
          subtitle={overview ? `$${overview.open_exposure.toFixed(0)} exposed` : undefined}
        />
        <MetricCard
          title="Total Wagered"
          value={overview ? formatCurrency(overview.total_wagered) : '$0'}
        />
      </div>

      {/* 2. Live Markets */}
      <LiveMarketsTable markets={liveMarkets} />

      {/* 3. Open Positions */}
      <OpenPositions bets={openBets} />

      {/* 4. Recent Bets */}
      <RecentBets bets={history} />

      {/* 5. Model Calibration */}
      <ModelCalibration buckets={calibration} />

      {/* 6. P&L by Asset */}
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

      {/* 7. Skip Log */}
      <SkipLog skips={skips} />
    </PageShell>
  );
}

/* -- Live Markets --------------------------------------------------- */

function LiveMarketsTable({ markets }: { markets: LiveMarket[] }) {
  const active = markets.filter(m => (m.minutes_left ?? 0) > 0 && !m.resolved);
  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        Live Markets ({active.length})
      </h3>
      {active.length === 0 ? (
        <div className="text-center text-gray-500 text-xs py-4">
          No live 15m markets found.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="text-left py-2 px-2">Asset</th>
                <th className="text-right py-2 px-2">YES</th>
                <th className="text-right py-2 px-2">NO</th>
                <th className="text-right py-2 px-2">Mins Left</th>
                <th className="text-right py-2 px-2">Volume</th>
                <th className="text-left py-2 px-2">Our Bet</th>
              </tr>
            </thead>
            <tbody>
              {active.map((m) => (
                <tr key={m.slug} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                  <td className="py-2 px-2 text-gray-200 font-medium">{m.asset}</td>
                  <td className="py-2 px-2 text-right text-accent-green">${m.yes_price.toFixed(3)}</td>
                  <td className="py-2 px-2 text-right text-accent-red">${m.no_price.toFixed(3)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">
                    {m.minutes_left != null ? `${m.minutes_left.toFixed(1)}m` : '--'}
                  </td>
                  <td className="py-2 px-2 text-right text-gray-400">
                    ${m.volume_24h.toFixed(0)}
                  </td>
                  <td className="py-2 px-2">
                    {m.our_bet ? (
                      <span className="px-1.5 py-0.5 rounded text-[10px] bg-accent-blue/20 text-accent-blue">
                        {m.our_bet.entry_side} ${m.our_bet.total_invested.toFixed(0)}
                      </span>
                    ) : (
                      <span className="text-gray-600">--</span>
                    )}
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

/* -- Open Positions ------------------------------------------------- */

function OpenPositions({ bets }: { bets: Bet[] }) {
  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        Open Positions ({bets.length})
      </h3>
      {bets.length === 0 ? (
        <div className="text-center text-gray-500 text-xs py-4">
          No open bets. Waiting for ML confidence &ge; 85%.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-500 border-b border-surface-3">
                <th className="text-left py-2 px-2">Asset</th>
                <th className="text-left py-2 px-2">Side</th>
                <th className="text-right py-2 px-2">Avg Cost</th>
                <th className="text-right py-2 px-2">Invested</th>
                <th className="text-right py-2 px-2">Tokens</th>
                <th className="text-center py-2 px-2">Adds</th>
                <th className="text-right py-2 px-2">Conf</th>
                <th className="text-left py-2 px-2">Age</th>
              </tr>
            </thead>
            <tbody>
              {bets.map((b) => (
                <tr key={b.id} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                  <td className="py-2 px-2 text-gray-200 font-medium">{b.asset}</td>
                  <td className="py-2 px-2">
                    <span className={b.entry_side === 'YES' ? 'text-accent-green' : 'text-accent-red'}>
                      {b.entry_side}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">${b.avg_cost.toFixed(3)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">${b.total_invested.toFixed(0)}</td>
                  <td className="py-2 px-2 text-right text-gray-400">{b.total_tokens.toFixed(1)}</td>
                  <td className="py-2 px-2 text-center text-gray-400">{b.adds_count}</td>
                  <td className="py-2 px-2 text-right text-gray-300">{b.entry_confidence.toFixed(0)}%</td>
                  <td className="py-2 px-2 text-gray-500">{formatTimestamp(b.opened_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* -- Recent Bets ---------------------------------------------------- */

function RecentBets({ bets }: { bets: Bet[] }) {
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
                <th className="text-left py-2 px-2">Side</th>
                <th className="text-right py-2 px-2">Avg Cost</th>
                <th className="text-right py-2 px-2">Invested</th>
                <th className="text-right py-2 px-2">P&L</th>
                <th className="text-right py-2 px-2">Return</th>
                <th className="text-left py-2 px-2">Result</th>
                <th className="text-left py-2 px-2">Reason</th>
              </tr>
            </thead>
            <tbody>
              {bets.map((b) => (
                <tr key={b.id} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                  <td className="py-2 px-2 text-gray-200 font-medium">{b.asset}</td>
                  <td className="py-2 px-2">
                    <span className={b.entry_side === 'YES' ? 'text-accent-green' : 'text-accent-red'}>
                      {b.entry_side}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">${b.avg_cost.toFixed(3)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">${b.total_invested.toFixed(0)}</td>
                  <td className={`py-2 px-2 text-right font-medium ${pnlColor(b.pnl ?? 0)}`}>
                    {b.pnl != null ? `$${b.pnl.toFixed(2)}` : '--'}
                  </td>
                  <td className={`py-2 px-2 text-right ${pnlColor(b.return_pct ?? 0)}`}>
                    {b.return_pct != null ? `${b.return_pct.toFixed(1)}%` : '--'}
                  </td>
                  <td className="py-2 px-2">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                      b.status === 'WON' ? 'bg-accent-green/20 text-accent-green' :
                      b.status === 'LOST' ? 'bg-accent-red/20 text-accent-red' :
                      b.status === 'CLOSED' ? 'bg-accent-blue/20 text-accent-blue' :
                      'bg-surface-3 text-gray-400'
                    }`}>
                      {b.status}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-gray-500 truncate max-w-[120px]" title={b.exit_reason ?? ''}>
                    {b.exit_reason ?? '--'}
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

/* -- Model Calibration ---------------------------------------------- */

function ModelCalibration({ buckets }: { buckets: CalibrationBucket[] }) {
  if (buckets.length === 0) return null;
  const hasData = buckets.some((b) => b.total > 0);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Model Calibration</h3>
      {!hasData ? (
        <div className="text-center text-gray-500 text-xs py-4">
          No resolved bets yet for calibration.
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-3">
          {buckets.map((b) => (
            <div key={b.label} className="bg-surface-2 rounded-lg p-3 text-center">
              <div className="text-xs text-gray-500 mb-1">{b.label}</div>
              <div className={`text-xl font-mono font-bold ${
                b.accuracy >= 60 ? 'text-accent-green' :
                b.accuracy >= 50 ? 'text-gray-300' :
                'text-accent-red'
              }`}>
                {b.total > 0 ? `${b.accuracy}%` : '--'}
              </div>
              <div className="text-[10px] text-gray-500 mt-1">
                {b.wins}W / {b.total - b.wins}L ({b.total} bets)
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* -- Skip Log ------------------------------------------------------- */

function SkipLog({ skips }: { skips: SkipEntry[] }) {
  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        Skip Log ({skips.length})
      </h3>
      {skips.length === 0 ? (
        <div className="text-center text-gray-500 text-xs py-4">
          No skipped opportunities yet.
        </div>
      ) : (
        <div className="space-y-1 max-h-[300px] overflow-y-auto">
          {skips.map((s, i) => (
            <div key={i} className="flex items-center gap-2 text-xs font-mono py-1 border-b border-surface-3/30">
              <span className="text-gray-500 w-16 shrink-0">{formatTimestamp(s.timestamp)}</span>
              <span className="text-gray-200 font-medium w-10">{s.asset}</span>
              <span className={s.ml_direction === 'UP' ? 'text-accent-green w-8' : 'text-accent-red w-8'}>
                {s.ml_direction}
              </span>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-accent-yellow/15 text-accent-yellow truncate max-w-[200px]">
                {s.reason}
              </span>
              <span className="text-gray-400">
                conf={s.ml_confidence?.toFixed(0) ?? '?'}%
              </span>
              {s.token_cost > 0 && (
                <span className="text-gray-500">
                  token=${s.token_cost.toFixed(2)}
                </span>
              )}
              {s.minutes_left > 0 && (
                <span className="text-gray-600">
                  {s.minutes_left.toFixed(0)}m left
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
