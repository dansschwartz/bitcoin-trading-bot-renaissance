import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import { api } from '../api';

interface AgentStatus {
  name: string;
  state: string;
  run_count?: number;
  error_count?: number;
  last_run_ts?: number;
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

export default function Agents() {
  const [statuses, setStatuses] = useState<AgentStatus[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [latestReport, setLatestReport] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    const load = () => {
      api.agentStatuses().then((d) => setStatuses(d as AgentStatus[])).catch(() => {});
      api.agentEvents(50).then((d) => setEvents(d as AgentEvent[])).catch(() => {});
      api.agentProposals(undefined, 20).then((d) => setProposals(d as Proposal[])).catch(() => {});
      api.agentLatestReport().then(setLatestReport).catch(() => {});
    };
    load();
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <PageShell title="Agent Coordination" subtitle="Doc 15 — autonomous agents, proposals, and weekly research">
      {/* Agent Status Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        {statuses.map((agent) => (
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
              {agent.run_count ?? agent.event_count ?? 0}
            </div>
            <div className="text-xs text-gray-500">
              {agent.error_count ? `${agent.error_count} errors` : 'runs'}
            </div>
          </div>
        ))}
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
                {latestReport.total_trades ?? '--'}
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
    </PageShell>
  );
}
