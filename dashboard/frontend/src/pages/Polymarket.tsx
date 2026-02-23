import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import MetricCard from '../components/cards/MetricCard';
import { api } from '../api';
import { formatCurrency, formatTimestamp, pnlColor } from '../utils/formatters';

interface PolymarketSummary {
  last_scan: string | null;
  total_markets: number;
  markets_by_type: Record<string, number>;
  opportunities: number;
  max_edge: number | null;
  top_opportunity: {
    asset: string;
    direction: string;
    edge: number;
    confidence: number;
    yes_price: number;
    question: string;
  } | null;
  error?: string;
}

interface EdgeOpportunity {
  condition_id: string;
  question: string;
  slug: string;
  market_type: string;
  asset: string;
  timeframe_minutes: number | null;
  deadline: string | null;
  target_price: number | null;
  yes_price: number;
  no_price: number;
  volume_24h: number;
  liquidity: number;
  edge: number;
  our_probability: number;
  direction: string;
  confidence: number;
}

interface BridgeSignal {
  source: string;
  direction: string;
  confidence: number;
  rawScore: number;
  agreement: number;
  regime: string;
  regimeAligned: boolean;
  breakoutScore: number;
  btcPrice: number;
  skipReason: string | null;
  observationMode: boolean;
  timestamp: string;
  scannerOpportunities?: EdgeOpportunity[];
  meta?: {
    activeSignals: number;
    agreeingSignals: number;
    disagreingSignals: number;
    modelConfidences: Record<string, number>;
  };
}

const TYPE_COLORS: Record<string, string> = {
  DIRECTION: 'bg-accent-blue/20 text-accent-blue',
  THRESHOLD: 'bg-purple-400/20 text-purple-400',
  HIT_PRICE: 'bg-accent-green/20 text-accent-green',
  RANGE: 'bg-accent-yellow/20 text-accent-yellow',
  VOLATILITY: 'bg-orange-400/20 text-orange-400',
  OTHER: 'bg-surface-3 text-gray-400',
};

export default function Polymarket() {
  const [summary, setSummary] = useState<PolymarketSummary | null>(null);
  const [edges, setEdges] = useState<EdgeOpportunity[]>([]);
  const [signal, setSignal] = useState<BridgeSignal | null>(null);
  const [tab, setTab] = useState<'edges' | 'markets' | 'signal'>('edges');
  const [markets, setMarkets] = useState<EdgeOpportunity[]>([]);
  const [typeFilter, setTypeFilter] = useState<string | null>(null);

  useEffect(() => {
    const load = () => {
      api.polymarketSummary().then((d) => setSummary(d as unknown as PolymarketSummary)).catch(() => {});
      api.polymarketEdges(0, 50).then((d) =>
        setEdges(((d as Record<string, unknown>).edges ?? []) as EdgeOpportunity[])
      ).catch(() => {});
      api.polymarketSignal().then((d) =>
        setSignal(((d as Record<string, unknown>).signal ?? null) as BridgeSignal | null)
      ).catch(() => {});
    };
    load();
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  }, []);

  // Load markets tab on demand
  useEffect(() => {
    if (tab === 'markets') {
      api.polymarketMarkets(typeFilter ?? undefined, undefined, 100)
        .then((d) =>
          setMarkets(((d as Record<string, unknown>).markets ?? []) as EdgeOpportunity[])
        )
        .catch(() => {});
    }
  }, [tab, typeFilter]);

  const topEdge = summary?.top_opportunity;

  return (
    <PageShell
      title="Polymarket"
      subtitle="Prediction market scanner + ML edge detection"
      actions={
        <div className="flex items-center gap-2">
          {signal?.observationMode && (
            <span className="px-2 py-1 rounded text-xs font-medium bg-accent-yellow/20 text-accent-yellow">
              OBSERVATION
            </span>
          )}
          <span className="text-xs text-gray-500">
            {summary?.last_scan
              ? `Scan: ${formatTimestamp(summary.last_scan)}`
              : 'No scans yet'}
          </span>
        </div>
      }
    >
      {/* Summary Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        <MetricCard
          title="Markets Scanned"
          value={summary?.total_markets ?? 0}
        />
        <MetricCard
          title="Opportunities"
          value={summary?.opportunities ?? 0}
          valueColor={
            (summary?.opportunities ?? 0) > 0 ? 'text-accent-green' : 'text-gray-400'
          }
        />
        <MetricCard
          title="Max Edge"
          value={
            summary?.max_edge != null
              ? `${(summary.max_edge * 100).toFixed(1)}%`
              : '--'
          }
          valueColor="text-accent-green"
        />
        <MetricCard
          title="ML Signal"
          value={signal ? `${signal.direction}` : '--'}
          subtitle={
            signal
              ? `conf ${signal.confidence}% | ${signal.regime}`
              : undefined
          }
          valueColor={
            signal?.direction === 'UP' ? 'text-accent-green' :
            signal?.direction === 'DOWN' ? 'text-accent-red' :
            'text-gray-400'
          }
        />
        <MetricCard
          title="BTC Price"
          value={signal?.btcPrice ? formatCurrency(signal.btcPrice) : '--'}
        />
      </div>

      {/* Market Type Distribution */}
      {summary?.markets_by_type && Object.keys(summary.markets_by_type).length > 0 && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Market Types</h3>
          <div className="flex flex-wrap gap-2">
            {Object.entries(summary.markets_by_type)
              .sort(([, a], [, b]) => b - a)
              .map(([type, count]) => (
                <div
                  key={type}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono cursor-pointer transition-opacity ${
                    TYPE_COLORS[type] || 'bg-surface-3 text-gray-400'
                  } ${typeFilter === type ? 'ring-1 ring-white/30' : 'opacity-80 hover:opacity-100'}`}
                  onClick={() => {
                    setTypeFilter(typeFilter === type ? null : type);
                    setTab('markets');
                  }}
                >
                  {type}: {count}
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Tab Switcher */}
      <div className="flex gap-1 bg-surface-1 border border-surface-3 rounded-lg p-1 w-fit">
        {(['edges', 'markets', 'signal'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              tab === t
                ? 'bg-accent-blue/20 text-accent-blue'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {t === 'edges' ? `Edge Opportunities (${edges.length})` :
             t === 'markets' ? 'All Markets' :
             'Bridge Signal'}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {tab === 'edges' && <EdgesTable edges={edges} />}
      {tab === 'markets' && <MarketsTable markets={markets} />}
      {tab === 'signal' && <SignalDetail signal={signal} />}
    </PageShell>
  );
}

function EdgesTable({ edges }: { edges: EdgeOpportunity[] }) {
  if (edges.length === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-6 text-center text-gray-500 text-sm">
        No edge opportunities detected in latest scan.
      </div>
    );
  }

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        Edge Opportunities — Ranked by Edge
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-gray-500 border-b border-surface-3">
              <th className="text-left py-2 px-2">Asset</th>
              <th className="text-left py-2 px-2">Type</th>
              <th className="text-left py-2 px-2">Direction</th>
              <th className="text-right py-2 px-2">Edge</th>
              <th className="text-right py-2 px-2">Our Prob</th>
              <th className="text-right py-2 px-2">Mkt Price</th>
              <th className="text-right py-2 px-2">Conf</th>
              <th className="text-left py-2 px-2">Market</th>
            </tr>
          </thead>
          <tbody>
            {edges.map((e, i) => (
              <tr key={e.condition_id || i} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                <td className="py-2 px-2 text-gray-200 font-medium">{e.asset}</td>
                <td className="py-2 px-2">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] ${TYPE_COLORS[e.market_type] || 'bg-surface-3 text-gray-400'}`}>
                    {e.market_type}
                  </span>
                </td>
                <td className="py-2 px-2">
                  <span className={e.direction === 'YES' || e.direction === 'UP'
                    ? 'text-accent-green' : 'text-accent-red'}>
                    {e.direction}
                  </span>
                </td>
                <td className="py-2 px-2 text-right text-accent-green font-medium">
                  {(e.edge * 100).toFixed(1)}%
                </td>
                <td className="py-2 px-2 text-right text-gray-300">
                  {(e.our_probability * 100).toFixed(1)}%
                </td>
                <td className="py-2 px-2 text-right text-gray-400">
                  {(e.yes_price * 100).toFixed(1)}c
                </td>
                <td className="py-2 px-2 text-right text-gray-300">
                  {e.confidence?.toFixed(0) ?? '--'}
                </td>
                <td className="py-2 px-2 text-gray-500 max-w-xs truncate" title={e.question}>
                  {truncateQuestion(e.question)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MarketsTable({ markets }: { markets: EdgeOpportunity[] }) {
  if (markets.length === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-6 text-center text-gray-500 text-sm">
        No markets found. Scanner may not have run yet.
      </div>
    );
  }

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        All Scanned Markets ({markets.length})
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-gray-500 border-b border-surface-3">
              <th className="text-left py-2 px-2">Asset</th>
              <th className="text-left py-2 px-2">Type</th>
              <th className="text-left py-2 px-2">Question</th>
              <th className="text-right py-2 px-2">YES</th>
              <th className="text-right py-2 px-2">NO</th>
              <th className="text-right py-2 px-2">Volume 24h</th>
              <th className="text-right py-2 px-2">Edge</th>
            </tr>
          </thead>
          <tbody>
            {markets.map((m, i) => (
              <tr key={m.condition_id || i} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                <td className="py-2 px-2 text-gray-200">{m.asset ?? '--'}</td>
                <td className="py-2 px-2">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] ${TYPE_COLORS[m.market_type] || 'bg-surface-3 text-gray-400'}`}>
                    {m.market_type}
                  </span>
                </td>
                <td className="py-2 px-2 text-gray-400 max-w-sm truncate" title={m.question}>
                  {truncateQuestion(m.question)}
                </td>
                <td className="py-2 px-2 text-right text-gray-300">
                  {m.yes_price != null ? `${(m.yes_price * 100).toFixed(1)}c` : '--'}
                </td>
                <td className="py-2 px-2 text-right text-gray-300">
                  {m.no_price != null ? `${(m.no_price * 100).toFixed(1)}c` : '--'}
                </td>
                <td className="py-2 px-2 text-right text-gray-400">
                  {m.volume_24h ? formatCurrency(m.volume_24h) : '--'}
                </td>
                <td className={`py-2 px-2 text-right ${m.edge && m.edge > 0 ? 'text-accent-green' : 'text-gray-500'}`}>
                  {m.edge != null && m.edge > 0 ? `${(m.edge * 100).toFixed(1)}%` : '--'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SignalDetail({ signal }: { signal: BridgeSignal | null }) {
  if (!signal) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-6 text-center text-gray-500 text-sm">
        No bridge signal available. Bot may not have generated one yet.
      </div>
    );
  }

  const mc = signal.meta?.modelConfidences ?? {};

  return (
    <div className="space-y-4">
      {/* Signal Overview */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Latest Bridge Signal</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <div className="text-xs text-gray-500">Direction</div>
            <div className={`text-2xl font-bold ${
              signal.direction === 'UP' ? 'text-accent-green' : 'text-accent-red'
            }`}>
              {signal.direction}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Confidence</div>
            <div className="text-2xl font-mono text-gray-200">{signal.confidence}%</div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Raw Score</div>
            <div className={`text-2xl font-mono ${pnlColor(signal.rawScore)}`}>
              {signal.rawScore >= 0 ? '+' : ''}{signal.rawScore.toFixed(4)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Agreement</div>
            <div className="text-2xl font-mono text-gray-200">
              {(signal.agreement * 100).toFixed(0)}%
            </div>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          <span className="px-2 py-1 rounded bg-surface-2 text-gray-400">
            Regime: {signal.regime}
          </span>
          <span className={`px-2 py-1 rounded ${
            signal.regimeAligned
              ? 'bg-accent-green/20 text-accent-green'
              : 'bg-surface-2 text-gray-400'
          }`}>
            {signal.regimeAligned ? 'Regime Aligned' : 'Not Aligned'}
          </span>
          {signal.skipReason && (
            <span className="px-2 py-1 rounded bg-accent-red/20 text-accent-red">
              SKIP: {signal.skipReason}
            </span>
          )}
          <span className="px-2 py-1 rounded bg-surface-2 text-gray-500">
            {formatTimestamp(signal.timestamp)}
          </span>
        </div>
      </div>

      {/* Model Confidences */}
      {Object.keys(mc).length > 0 && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Per-Model Predictions</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {Object.entries(mc)
              .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
              .map(([model, pred]) => (
                <div key={model} className="bg-surface-2 rounded-lg p-2">
                  <div className="text-[10px] text-gray-500 truncate">{model}</div>
                  <div className={`text-sm font-mono ${pnlColor(pred)}`}>
                    {pred >= 0 ? '+' : ''}{pred.toFixed(4)}
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

function truncateQuestion(q: string | null | undefined): string {
  if (!q) return '--';
  return q.length > 60 ? q.slice(0, 57) + '...' : q;
}
