import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import MetricCard from '../components/cards/MetricCard';
import { api } from '../api';
import { formatTimestamp, formatNumber } from '../utils/formatters';

interface BreakoutSummary {
  total_scans: number;
  total_flagged: number;
  avg_flagged_per_scan: number;
  last_scan_seconds: number;
  pairs_tracked: number;
  last_scan_time: string | null;
}

interface BreakoutSignal {
  symbol: string;
  score: number;
  signal_strength: number;
  volume_surge: number;
  momentum: number;
  timestamp: string;
}

interface BreakoutSignalsResponse {
  scan_time: string | null;
  total_scanned: number;
  total_flagged: number;
  signals: BreakoutSignal[];
}

interface HeatmapPair {
  symbol: string;
  score: number;
  tier: string;
}

interface BreakoutHistoryEntry {
  product_id: string;
  score: number;
  signal_strength: number;
  reasoning: string;
  timestamp: string;
}

export default function BreakoutScanner() {
  const [summary, setSummary] = useState<BreakoutSummary | null>(null);
  const [signals, setSignals] = useState<BreakoutSignalsResponse | null>(null);
  const [heatmap, setHeatmap] = useState<HeatmapPair[]>([]);
  const [history, setHistory] = useState<BreakoutHistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => {
    const load = () => {
      api.breakoutSummary().then((d) => setSummary(d as unknown as BreakoutSummary)).catch(() => {});
      api.breakoutSignals(30).then((d) => setSignals(d as unknown as BreakoutSignalsResponse)).catch(() => {});
      api.breakoutHeatmap().then((d) =>
        setHeatmap(((d as Record<string, unknown>).pairs ?? []) as HeatmapPair[])
      ).catch(() => {});
    };
    load();
    const id = setInterval(load, 15_000);
    return () => clearInterval(id);
  }, []);

  const flaggedSignals = signals?.signals ?? [];

  return (
    <PageShell
      title="Breakout Scanner"
      subtitle="Real-time breakout detection across the trading universe"
      actions={
        <span className="text-xs text-gray-500">
          {summary?.last_scan_time
            ? `Last scan: ${formatTimestamp(summary.last_scan_time)}`
            : 'Waiting for first scan...'}
        </span>
      }
    >
      {/* Summary Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        <MetricCard
          title="Total Scans"
          value={summary?.total_scans ?? 0}
        />
        <MetricCard
          title="Pairs Tracked"
          value={summary?.pairs_tracked ?? 0}
        />
        <MetricCard
          title="Flagged This Scan"
          value={signals?.total_flagged ?? 0}
          subtitle={`of ${signals?.total_scanned ?? 0} scanned`}
          valueColor={
            (signals?.total_flagged ?? 0) > 0
              ? 'text-accent-yellow'
              : 'text-gray-400'
          }
        />
        <MetricCard
          title="Total Flagged"
          value={summary?.total_flagged ?? 0}
          subtitle={`avg ${(summary?.avg_flagged_per_scan ?? 0).toFixed(1)}/scan`}
        />
        <MetricCard
          title="Scan Time"
          value={`${(summary?.last_scan_seconds ?? 0).toFixed(1)}s`}
        />
      </div>

      {/* Active Breakout Signals */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">
          Active Breakout Signals
          {flaggedSignals.length > 0 && (
            <span className="ml-2 px-2 py-0.5 rounded bg-accent-yellow/20 text-accent-yellow text-xs">
              {flaggedSignals.length}
            </span>
          )}
        </h3>
        {flaggedSignals.length === 0 ? (
          <p className="text-sm text-gray-500">No breakout signals detected in latest scan.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-gray-500 border-b border-surface-3">
                  <th className="text-left py-2 px-2">Symbol</th>
                  <th className="text-right py-2 px-2">Score</th>
                  <th className="text-right py-2 px-2">Signal Str.</th>
                  <th className="text-right py-2 px-2">Vol Surge</th>
                  <th className="text-right py-2 px-2">Momentum</th>
                  <th className="text-left py-2 px-2">Time</th>
                </tr>
              </thead>
              <tbody>
                {flaggedSignals.map((s, i) => (
                  <tr key={i} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                    <td className="py-2 px-2 text-gray-200 font-medium">{s.symbol}</td>
                    <td className="py-2 px-2 text-right">
                      <span className={scoreColor(s.score)}>
                        {s.score?.toFixed(1) ?? '--'}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right text-gray-300">
                      {s.signal_strength?.toFixed(3) ?? '--'}
                    </td>
                    <td className="py-2 px-2 text-right text-gray-300">
                      {s.volume_surge != null ? `${s.volume_surge.toFixed(1)}x` : '--'}
                    </td>
                    <td className="py-2 px-2 text-right text-gray-300">
                      {s.momentum?.toFixed(3) ?? '--'}
                    </td>
                    <td className="py-2 px-2 text-gray-500">
                      {s.timestamp ? formatTimestamp(s.timestamp) : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Heatmap */}
      {heatmap.length > 0 && (
        <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">
            Score Heatmap — Top {heatmap.length} Pairs
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {heatmap.map((p) => (
              <div
                key={p.symbol}
                className={`px-2 py-1 rounded text-[10px] font-mono ${heatmapColor(p.score)}`}
                title={`${p.symbol}: ${p.score.toFixed(1)} (${p.tier})`}
              >
                {p.symbol.replace('-USD', '').replace('USDT', '')}
                <span className="ml-1 opacity-75">{p.score.toFixed(0)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* History (collapsible) */}
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <button
          onClick={() => {
            if (!showHistory && history.length === 0) {
              api.breakoutHistory(24, 100)
                .then((d: Record<string, unknown>) =>
                  setHistory((d.entries ?? []) as BreakoutHistoryEntry[])
                )
                .catch(() => {});
            }
            setShowHistory(!showHistory);
          }}
          className="flex items-center gap-2 text-sm font-medium text-gray-300 hover:text-gray-100"
        >
          <svg
            className={`w-4 h-4 transition-transform ${showHistory ? 'rotate-90' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          24h Flag History
        </button>
        {showHistory && (
          <div className="overflow-x-auto mt-3">
            {history.length === 0 ? (
              <p className="text-xs text-gray-500">No breakout flags in the last 24 hours.</p>
            ) : (
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-gray-500 border-b border-surface-3">
                    <th className="text-left py-2 px-2">Time</th>
                    <th className="text-left py-2 px-2">Symbol</th>
                    <th className="text-right py-2 px-2">Score</th>
                    <th className="text-right py-2 px-2">Signal</th>
                    <th className="text-left py-2 px-2">Reasoning</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((h, i) => (
                    <tr key={i} className="border-b border-surface-3/50">
                      <td className="py-2 px-2 text-gray-500">{formatTimestamp(h.timestamp)}</td>
                      <td className="py-2 px-2 text-gray-300">{h.product_id}</td>
                      <td className="py-2 px-2 text-right">
                        <span className={scoreColor(h.score)}>{h.score?.toFixed(1) ?? '--'}</span>
                      </td>
                      <td className="py-2 px-2 text-right text-gray-300">
                        {h.signal_strength?.toFixed(3) ?? '--'}
                      </td>
                      <td className="py-2 px-2 text-gray-500 max-w-xs truncate">
                        {h.reasoning ?? '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </div>
    </PageShell>
  );
}

function scoreColor(score: number): string {
  if (score >= 70) return 'text-accent-green';
  if (score >= 50) return 'text-accent-yellow';
  if (score >= 30) return 'text-orange-400';
  return 'text-gray-400';
}

function heatmapColor(score: number): string {
  if (score >= 70) return 'bg-accent-green/30 text-accent-green';
  if (score >= 50) return 'bg-accent-yellow/30 text-accent-yellow';
  if (score >= 30) return 'bg-orange-400/20 text-orange-400';
  return 'bg-surface-2 text-gray-500';
}
