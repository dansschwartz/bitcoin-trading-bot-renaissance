import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import { api } from '../api';

interface AgentStatus {
  name: string;
  state: string;
  run_count?: number;
  event_count?: number;
  error_count?: number;
  last_run_ts?: number;
  last_event?: string;
  [key: string]: unknown;
}

interface AgentEvent {
  id: number;
  timestamp: string;
  agent_name: string;
  event_type: string;
  channel: string;
  severity: string;
  payload?: Record<string, unknown>;
}

interface Proposal {
  id: number;
  title: string;
  category: string;
  status: string;
  deployment_mode: string;
  backtest_sharpe?: number;
  backtest_accuracy?: number;
  created_at: string;
  safety_gate_results?: Record<string, unknown>;
}

interface ModelLedgerEntry {
  id: number;
  model_name: string;
  model_version: string;
  accuracy?: number;
  sharpe?: number;
  status: string;
  file_hash?: string;
  timestamp?: string;
  [key: string]: unknown;
}

interface ImprovementEntry {
  id: number;
  timestamp: string;
  change_type: string;
  description: string;
  metric_before?: number;
  metric_after?: number;
  metric_name?: string;
  proposal_id?: number;
  [key: string]: unknown;
}

const SEVERITY_COLORS: Record<string, string> = {
  info: 'text-accent-blue',
  warning: 'text-accent-yellow',
  error: 'text-accent-red',
};

const STATUS_COLORS: Record<string, string> = {
  pending: 'bg-gray-600',
  safety_passed: 'bg-accent-green/20 text-accent-green',
  safety_failed: 'bg-accent-red/20 text-accent-red',
  sandboxing: 'bg-accent-yellow/20 text-accent-yellow',
  deployed: 'bg-accent-blue/20 text-accent-blue',
  rolled_back: 'bg-gray-500/20 text-gray-400',
};

const CHANGE_TYPE_COLORS: Record<string, string> = {
  parameter_tune: 'border-accent-green text-accent-green',
  modify_existing: 'border-accent-blue text-accent-blue',
  new_feature: 'border-purple-400 text-purple-400',
};

const CHANGE_TYPE_BG: Record<string, string> = {
  parameter_tune: 'bg-accent-green/10',
  modify_existing: 'bg-accent-blue/10',
  new_feature: 'bg-purple-400/10',
};

function formatRelativeTime(dateStr: string | undefined | null): string {
  if (!dateStr) return '--';
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

function accuracyColor(acc: number | undefined | null): string {
  if (acc == null) return 'text-gray-400';
  if (acc > 0.52) return 'text-accent-green';
  if (acc >= 0.50) return 'text-accent-yellow';
  return 'text-accent-red';
}

function truncateHash(hash: string | undefined | null): string {
  if (!hash) return '--';
  if (hash.length <= 12) return hash;
  return hash.slice(0, 8) + '...';
}

export default function Agents() {
  const [statuses, setStatuses] = useState<AgentStatus[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [latestReport, setLatestReport] = useState<Record<string, unknown> | null>(null);
  const [models, setModels] = useState<ModelLedgerEntry[]>([]);
  const [improvements, setImprovements] = useState<ImprovementEntry[]>([]);

  useEffect(() => {
    const load = () => {
      api.agentStatuses().then((d) => setStatuses(d as AgentStatus[])).catch(() => {});
      api.agentEvents(50).then((d) => setEvents(d as unknown as AgentEvent[])).catch(() => {});
      api.agentProposals(undefined, 20).then((d) => setProposals(d as unknown as Proposal[])).catch(() => {});
      api.agentLatestReport().then(setLatestReport).catch(() => {});
      api.agentModels().then((d) => setModels(d as unknown as ModelLedgerEntry[])).catch(() => {});
      api.agentImprovements(30).then((d) => setImprovements(d as unknown as ImprovementEntry[])).catch(() => {});
    };
    load();
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <PageShell title="Agent Coordination">
      {/* Agent Status Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        {statuses.map((agent) => {
          const lastActive = agent.last_run_ts
            ? new Date(agent.last_run_ts * 1000).toISOString()
            : (agent.last_event as string | undefined);
          return (
            <div
              key={agent.name}
              className="bg-surface-1 border border-surface-3 rounded-lg p-3"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-gray-300 uppercase tracking-wide">
                  {agent.name}
                </span>
                <span
                  className={`w-2 h-2 rounded-full ${
                    agent.state === 'active'
                      ? 'bg-accent-green'
                      : agent.state === 'error'
                      ? 'bg-accent-red'
                      : 'bg-gray-500'
                  }`}
                />
              </div>
              <div className="text-lg font-mono text-gray-100">
                {String(agent.run_count ?? agent.event_count ?? 0)}
              </div>
              <div className="text-xs text-gray-500">
                {agent.error_count
                  ? <span className="text-accent-red font-medium">{agent.error_count} errors</span>
                  : 'runs'}
              </div>
              <div className="text-[10px] text-gray-600 mt-1 font-mono">
                {formatRelativeTime(lastActive)}
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Event Stream */}
        <div className="bg-surface-1 border border-surface-3 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Recent Events</h2>
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {events.length === 0 ? (
              <p className="text-xs text-gray-500">No agent events yet</p>
            ) : (
              events.map((ev) => (
                <div
                  key={ev.id}
                  className="flex items-start gap-2 text-xs py-1 border-b border-surface-3 last:border-0"
                >
                  <span className="text-gray-500 font-mono shrink-0 w-16">
                    {new Date(ev.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="text-gray-400 shrink-0 w-14">{ev.agent_name}</span>
                  <span className={SEVERITY_COLORS[ev.severity] || 'text-gray-300'}>
                    {ev.event_type}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Proposals */}
        <div className="bg-surface-1 border border-surface-3 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Proposals</h2>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {proposals.length === 0 ? (
              <p className="text-xs text-gray-500">
                No proposals yet. Enable quant_researcher in config to generate weekly research.
              </p>
            ) : (
              proposals.map((p) => (
                <div
                  key={p.id}
                  className="flex items-center gap-2 text-xs py-2 px-2 bg-surface-2 rounded"
                >
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                      STATUS_COLORS[p.status] || 'bg-gray-600'
                    }`}
                  >
                    {p.status}
                  </span>
                  <span className="text-gray-200 flex-1 truncate">{p.title}</span>
                  <span className="text-gray-500">{p.category}</span>
                  {p.backtest_sharpe != null && (
                    <span className="text-gray-400 font-mono">
                      S:{p.backtest_sharpe.toFixed(2)}
                    </span>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Latest Report Summary */}
      {latestReport && !('error' in latestReport) && !('message' in latestReport) && (
        <div className="bg-surface-1 border border-surface-3 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Latest Weekly Report</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <div>
              <div className="text-xs text-gray-500">Generated</div>
              <div className="text-sm text-gray-200 font-mono">
                {latestReport.generated_at
                  ? new Date(latestReport.generated_at as string).toLocaleDateString()
                  : '--'}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">7d Sharpe</div>
              <div className="text-sm text-gray-200 font-mono">
                {latestReport.sharpe_7d != null
                  ? (latestReport.sharpe_7d as number).toFixed(3)
                  : '--'}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Total P&L</div>
              <div className="text-sm text-gray-200 font-mono">
                ${latestReport.total_pnl != null
                  ? (latestReport.total_pnl as number).toFixed(2)
                  : '--'}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Trades</div>
              <div className="text-sm text-gray-200 font-mono">
                {String(latestReport.total_trades ?? '--')}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Win Rate</div>
              <div className="text-sm text-gray-200 font-mono">
                {latestReport.win_rate != null
                  ? `${((latestReport.win_rate as number) * 100).toFixed(1)}%`
                  : '--'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Ledger */}
      <div className="bg-surface-1 border border-surface-3 rounded-lg p-4">
        <h2 className="text-sm font-semibold text-gray-300 mb-3">Model Ledger</h2>
        {models.length === 0 ? (
          <p className="text-xs text-gray-500">No model versions recorded yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-surface-3 text-gray-500">
                  <th className="text-left py-2 pr-3 font-medium">Model Name</th>
                  <th className="text-left py-2 pr-3 font-medium">Version</th>
                  <th className="text-right py-2 pr-3 font-medium">Accuracy</th>
                  <th className="text-right py-2 pr-3 font-medium">Sharpe</th>
                  <th className="text-center py-2 pr-3 font-medium">Status</th>
                  <th className="text-left py-2 pr-3 font-medium">File Hash</th>
                  <th className="text-right py-2 font-medium">Last Updated</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.id} className="border-b border-surface-3 last:border-0">
                    <td className="py-2 pr-3 text-gray-200 font-mono">{m.model_name}</td>
                    <td className="py-2 pr-3 text-gray-400 font-mono">{m.model_version}</td>
                    <td className={`py-2 pr-3 text-right font-mono ${accuracyColor(m.accuracy)}`}>
                      {m.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : '--'}
                    </td>
                    <td className="py-2 pr-3 text-right font-mono text-gray-300">
                      {m.sharpe != null ? m.sharpe.toFixed(3) : '--'}
                    </td>
                    <td className="py-2 pr-3 text-center">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                          m.status === 'active'
                            ? 'bg-accent-green/20 text-accent-green'
                            : 'bg-gray-600/40 text-gray-400'
                        }`}
                      >
                        {m.status}
                      </span>
                    </td>
                    <td className="py-2 pr-3 text-gray-500 font-mono" title={m.file_hash ?? ''}>
                      {truncateHash(m.file_hash)}
                    </td>
                    <td className="py-2 text-right text-gray-500 font-mono">
                      {m.timestamp
                        ? new Date(m.timestamp).toLocaleDateString()
                        : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Improvement Log Timeline */}
      <div className="bg-surface-1 border border-surface-3 rounded-lg p-4">
        <h2 className="text-sm font-semibold text-gray-300 mb-3">Improvement Log</h2>
        {improvements.length === 0 ? (
          <p className="text-xs text-gray-500">No improvements logged yet.</p>
        ) : (
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {improvements.map((imp) => {
              const typeColor = CHANGE_TYPE_COLORS[imp.change_type] || 'border-gray-500 text-gray-400';
              const typeBg = CHANGE_TYPE_BG[imp.change_type] || 'bg-gray-600/10';
              return (
                <div
                  key={imp.id}
                  className={`flex items-start gap-3 text-xs py-2 px-3 rounded border-l-2 ${typeColor} ${typeBg}`}
                >
                  <div className="shrink-0 w-20">
                    <div className="text-gray-500 font-mono">
                      {new Date(imp.timestamp).toLocaleDateString()}
                    </div>
                    <div className="text-gray-600 font-mono">
                      {new Date(imp.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                          imp.change_type === 'parameter_tune'
                            ? 'bg-accent-green/20 text-accent-green'
                            : imp.change_type === 'modify_existing'
                            ? 'bg-accent-blue/20 text-accent-blue'
                            : imp.change_type === 'new_feature'
                            ? 'bg-purple-400/20 text-purple-400'
                            : 'bg-gray-600 text-gray-400'
                        }`}
                      >
                        {imp.change_type}
                      </span>
                    </div>
                    <div className="text-gray-300 mt-0.5">{imp.description}</div>
                  </div>
                  {(imp.metric_before != null || imp.metric_after != null) && (
                    <div className="shrink-0 text-right">
                      <div className="text-gray-500">{imp.metric_name || 'metric'}</div>
                      <div className="font-mono">
                        <span className="text-gray-500">
                          {imp.metric_before != null ? imp.metric_before.toFixed(3) : '?'}
                        </span>
                        <span className="text-gray-600 mx-1">&rarr;</span>
                        <span
                          className={
                            imp.metric_after != null && imp.metric_before != null
                              ? imp.metric_after > imp.metric_before
                                ? 'text-accent-green'
                                : imp.metric_after < imp.metric_before
                                ? 'text-accent-red'
                                : 'text-gray-400'
                              : 'text-gray-400'
                          }
                        >
                          {imp.metric_after != null ? imp.metric_after.toFixed(3) : '?'}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </PageShell>
  );
}
