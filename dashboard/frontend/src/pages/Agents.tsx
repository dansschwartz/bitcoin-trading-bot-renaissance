import { useEffect, useState, useCallback } from 'react';
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

interface CouncilSession {
  session_id: string;
  timestamp: string;
  proposal_count: number;
  review_count: number;
  consensus_count: number;
}

interface CouncilProposal {
  title: string;
  description: string;
  category: string;
  consensus_score: number;
  endorsements: number;
  rejections: number;
  passes_consensus: boolean;
  review_count: number;
  _source_researcher: string;
  notes?: string;
  expected_improvement_bps?: number;
  [key: string]: unknown;
}

interface CouncilLatest {
  session_id: string | null;
  timestamp?: string;
  proposals: CouncilProposal[];
  stats: { total: number; consensus_passed: number; avg_score: number };
}

interface ResearcherInfo {
  name: string;
  proposals: Record<string, unknown>[];
  reviews: Record<string, unknown>[];
  proposal_count: number;
  review_count: number;
}

interface AuditRow {
  timestamp?: string;
  pair?: string;
  action?: string;
  blocked_by_gate?: string;
  regime_label?: string;
  confidence?: number;
  weighted_signal?: number;
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

const RESEARCHER_LABELS: Record<string, { color: string; short: string }> = {
  mathematician: { color: 'text-blue-400', short: 'MATH' },
  cryptographer: { color: 'text-purple-400', short: 'CRYPT' },
  physicist: { color: 'text-cyan-400', short: 'PHYS' },
  linguist: { color: 'text-amber-400', short: 'LING' },
  systems_engineer: { color: 'text-emerald-400', short: 'SYS' },
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
  // Existing agent state
  const [statuses, setStatuses] = useState<AgentStatus[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [latestReport, setLatestReport] = useState<Record<string, unknown> | null>(null);
  const [models, setModels] = useState<ModelLedgerEntry[]>([]);
  const [improvements, setImprovements] = useState<ImprovementEntry[]>([]);

  // Council state
  const [councilSessions, setCouncilSessions] = useState<CouncilSession[]>([]);
  const [councilLatest, setCouncilLatest] = useState<CouncilLatest | null>(null);
  const [selectedSession, setSelectedSession] = useState<string>('');
  const [expandedProposal, setExpandedProposal] = useState<number | null>(null);
  const [councilOpen, setCouncilOpen] = useState(true);

  // Researcher state
  const [expandedResearcher, setExpandedResearcher] = useState<string | null>(null);
  const [researcherInfo, setResearcherInfo] = useState<ResearcherInfo | null>(null);

  // Audit state
  const [auditOpen, setAuditOpen] = useState(false);
  const [auditSummary, setAuditSummary] = useState<Record<string, unknown> | null>(null);
  const [auditRecent, setAuditRecent] = useState<AuditRow[]>([]);

  // Load agent data
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

  // Load council data
  useEffect(() => {
    const loadCouncil = () => {
      api.councilSessions().then((d) => setCouncilSessions(d as unknown as CouncilSession[])).catch(() => {});
      api.councilLatest().then((d) => setCouncilLatest(d as unknown as CouncilLatest)).catch(() => {});
    };
    loadCouncil();
    const id = setInterval(loadCouncil, 60_000);
    return () => clearInterval(id);
  }, []);

  // Load audit data when section is opened
  useEffect(() => {
    if (!auditOpen) return;
    const loadAudit = () => {
      api.auditSummary(24).then(setAuditSummary).catch(() => {});
      api.auditRecent(20).then((d) => setAuditRecent(d as unknown as AuditRow[])).catch(() => {});
    };
    loadAudit();
    const id = setInterval(loadAudit, 60_000);
    return () => clearInterval(id);
  }, [auditOpen]);

  // Load researcher detail when expanded
  const loadResearcher = useCallback((name: string) => {
    const sessionId = councilLatest?.session_id;
    if (!sessionId) return;
    if (expandedResearcher === name) {
      setExpandedResearcher(null);
      setResearcherInfo(null);
      return;
    }
    setExpandedResearcher(name);
    api.councilResearcher(sessionId, name)
      .then((d) => setResearcherInfo(d as unknown as ResearcherInfo))
      .catch(() => setResearcherInfo(null));
  }, [councilLatest, expandedResearcher]);

  // Load different session when selected
  useEffect(() => {
    if (!selectedSession || selectedSession === councilLatest?.session_id) return;
    api.councilSession(selectedSession)
      .then((d) => {
        const data = d as unknown as CouncilLatest;
        setCouncilLatest({ ...data, session_id: selectedSession });
      })
      .catch(() => {});
  }, [selectedSession]);

  // Derive proposals to show from council latest
  const councilProposals = councilLatest?.proposals ?? [];

  // Derive gate blocks for audit chart
  const gateBlocks: { gate: string; count: number }[] = [];
  if (auditSummary && typeof auditSummary === 'object') {
    const gates = (auditSummary as Record<string, unknown>).gate_blocks as Record<string, number> | undefined;
    if (gates) {
      Object.entries(gates)
        .sort((a, b) => b[1] - a[1])
        .forEach(([gate, count]) => gateBlocks.push({ gate, count }));
    }
  }
  const maxGateCount = Math.max(1, ...gateBlocks.map((g) => g.count));

  // Derive funnel stats
  const totalDecisions = (auditSummary as Record<string, unknown>)?.total_decisions as number ?? 0;
  const passedGates = (auditSummary as Record<string, unknown>)?.passed_gates as number ?? 0;
  const totalTrades = (auditSummary as Record<string, unknown>)?.total_trades as number ?? 0;

  // Derive regime distribution
  const regimeDist: { label: string; count: number }[] = [];
  if (auditSummary) {
    const rd = (auditSummary as Record<string, unknown>).regime_distribution as Record<string, number> | undefined;
    if (rd) {
      Object.entries(rd)
        .sort((a, b) => b[1] - a[1])
        .forEach(([label, count]) => regimeDist.push({ label, count }));
    }
  }
  const totalRegime = regimeDist.reduce((s, r) => s + r.count, 0) || 1;

  // Researcher proposal counts from latest council
  const researcherCounts: Record<string, { proposals: number; topTitle: string }> = {};
  for (const p of councilProposals) {
    const r = p._source_researcher;
    if (!researcherCounts[r]) researcherCounts[r] = { proposals: 0, topTitle: '' };
    researcherCounts[r].proposals++;
    if (!researcherCounts[r].topTitle) researcherCounts[r].topTitle = p.title;
  }

  const REGIME_COLORS: Record<string, string> = {
    bull_trending: '#22c55e',
    bear_trending: '#ef4444',
    neutral_sideways: '#6b7280',
    bull_mean_reverting: '#3b82f6',
    bear_mean_reverting: '#f59e0b',
    unknown: '#4b5563',
  };

  return (
    <PageShell title="Intelligence Hub">
      {/* ═══════════════ EXISTING: Agent Status Grid ═══════════════ */}
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

      {/* ═══════════════ NEW: Research Council ═══════════════ */}
      <div className="bg-surface-1 border border-surface-3 rounded-lg">
        <button
          onClick={() => setCouncilOpen(!councilOpen)}
          className="w-full flex items-center justify-between p-4 text-left"
        >
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-gray-300">Research Council</h2>
            {councilLatest && councilLatest.session_id && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-accent-blue/20 text-accent-blue">
                {councilLatest.stats.total} proposals
              </span>
            )}
            {councilLatest && councilLatest.stats.consensus_passed > 0 && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-accent-green/20 text-accent-green">
                {councilLatest.stats.consensus_passed} consensus
              </span>
            )}
          </div>
          <svg
            className={`w-4 h-4 text-gray-500 transition-transform ${councilOpen ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {councilOpen && (
          <div className="px-4 pb-4 space-y-4">
            {/* Session selector */}
            {councilSessions.length > 1 && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">Session:</span>
                <select
                  className="bg-surface-2 border border-surface-3 rounded text-xs text-gray-300 px-2 py-1"
                  value={selectedSession || councilLatest?.session_id || ''}
                  onChange={(e) => setSelectedSession(e.target.value)}
                >
                  {councilSessions.map((s) => (
                    <option key={s.session_id} value={s.session_id}>
                      {new Date(s.timestamp).toLocaleString()} ({s.proposal_count} proposals, {s.consensus_count} consensus)
                    </option>
                  ))}
                </select>
              </div>
            )}

            {councilProposals.length === 0 ? (
              <p className="text-xs text-gray-500">
                No council sessions yet. Run: <code className="text-gray-400">.venv/bin/python3 -m agents.quant_researcher --council</code>
              </p>
            ) : (
              <div className="space-y-1">
                {/* Ranked proposals table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-surface-3 text-gray-500">
                        <th className="text-center py-2 pr-2 font-medium w-8">#</th>
                        <th className="text-left py-2 pr-3 font-medium">Title</th>
                        <th className="text-left py-2 pr-3 font-medium w-16">Researcher</th>
                        <th className="text-right py-2 pr-3 font-medium w-14">Score</th>
                        <th className="text-center py-2 pr-3 font-medium w-20">Endorsements</th>
                        <th className="text-center py-2 font-medium w-20">Consensus</th>
                      </tr>
                    </thead>
                    <tbody>
                      {councilProposals.map((p, idx) => {
                        const rLabel = RESEARCHER_LABELS[p._source_researcher] || { color: 'text-gray-400', short: p._source_researcher?.slice(0, 4)?.toUpperCase() || '?' };
                        const isExpanded = expandedProposal === idx;
                        return (
                          <tr key={idx} className="border-b border-surface-3 last:border-0">
                            <td colSpan={6} className="p-0">
                              <button
                                onClick={() => setExpandedProposal(isExpanded ? null : idx)}
                                className="w-full flex items-center text-left hover:bg-surface-2 transition-colors"
                              >
                                <span className="py-2 pr-2 text-center text-gray-500 font-mono w-8 shrink-0">
                                  {idx + 1}
                                </span>
                                <span className="py-2 pr-3 text-gray-200 flex-1 truncate">
                                  {p.title}
                                </span>
                                <span className={`py-2 pr-3 font-mono text-[10px] uppercase w-16 shrink-0 ${rLabel.color}`}>
                                  {rLabel.short}
                                </span>
                                <span className="py-2 pr-3 text-right font-mono text-gray-300 w-14 shrink-0">
                                  {p.consensus_score?.toFixed(1) ?? '--'}
                                </span>
                                <span className="py-2 pr-3 text-center font-mono text-gray-400 w-20 shrink-0">
                                  {p.endorsements}/{p.review_count}
                                </span>
                                <span className={`py-2 text-center w-20 shrink-0 ${
                                  p.passes_consensus ? 'text-accent-green' : 'text-gray-500'
                                }`}>
                                  {p.passes_consensus ? 'PASSED' : 'FAILED'}
                                </span>
                              </button>
                              {isExpanded && (
                                <div className="px-4 pb-3 pt-1 bg-surface-2 rounded-b">
                                  <div className="text-xs text-gray-400 mb-2">
                                    {p.description}
                                  </div>
                                  {p.notes && (
                                    <div className="text-[11px] text-gray-500 border-t border-surface-3 pt-2 mt-2">
                                      <span className="text-gray-400 font-medium">Notes: </span>
                                      {p.notes}
                                    </div>
                                  )}
                                  <div className="flex gap-4 mt-2 text-[10px] text-gray-500">
                                    <span>Category: <span className="text-gray-400">{p.category}</span></span>
                                    {p.expected_improvement_bps != null && (
                                      <span>Expected: <span className="text-accent-green">+{p.expected_improvement_bps}bps</span></span>
                                    )}
                                    <span>Rejections: <span className={p.rejections > 0 ? 'text-accent-red' : 'text-gray-400'}>{p.rejections}</span></span>
                                  </div>
                                </div>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Researcher Grid */}
            <div>
              <h3 className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wide">Researchers</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
                {Object.entries(RESEARCHER_LABELS).map(([name, meta]) => {
                  const counts = researcherCounts[name];
                  const isActive = expandedResearcher === name;
                  return (
                    <button
                      key={name}
                      onClick={() => loadResearcher(name)}
                      className={`text-left bg-surface-2 border rounded-lg p-3 transition-colors ${
                        isActive ? 'border-accent-blue' : 'border-surface-3 hover:border-gray-600'
                      }`}
                    >
                      <div className={`text-xs font-medium uppercase tracking-wide ${meta.color}`}>
                        {name.replace('_', ' ')}
                      </div>
                      <div className="mt-1 text-lg font-mono text-gray-100">
                        {counts?.proposals ?? 0}
                      </div>
                      <div className="text-[10px] text-gray-500">proposals</div>
                      {counts?.topTitle && (
                        <div className="text-[10px] text-gray-500 mt-1 truncate" title={counts.topTitle}>
                          {counts.topTitle}
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Expanded researcher detail */}
              {expandedResearcher && researcherInfo && (
                <div className="mt-3 bg-surface-2 border border-surface-3 rounded-lg p-4">
                  <h4 className="text-xs font-semibold text-gray-300 mb-2">
                    {expandedResearcher.replace('_', ' ')} — {researcherInfo.proposal_count} proposals, {researcherInfo.review_count} reviews
                  </h4>
                  {researcherInfo.proposals.length > 0 && (
                    <div className="mb-3">
                      <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">Proposals</div>
                      <div className="space-y-1">
                        {researcherInfo.proposals.map((p, i) => (
                          <div key={i} className="text-xs text-gray-300 py-1 border-b border-surface-3 last:border-0">
                            {(p as Record<string, unknown>).title as string || `Proposal ${i + 1}`}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {researcherInfo.reviews.length > 0 && (
                    <div>
                      <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">Reviews</div>
                      <div className="space-y-1 max-h-40 overflow-y-auto">
                        {researcherInfo.reviews.map((r, i) => {
                          const rev = r as Record<string, unknown>;
                          return (
                            <div key={i} className="flex items-center gap-2 text-xs py-1 border-b border-surface-3 last:border-0">
                              <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                                rev.verdict === 'endorse' ? 'bg-accent-green/20 text-accent-green'
                                  : rev.verdict === 'challenge' ? 'bg-accent-red/20 text-accent-red'
                                  : 'bg-gray-600 text-gray-400'
                              }`}>
                                {String(rev.verdict ?? '?')}
                              </span>
                              <span className="text-gray-400 truncate flex-1">
                                {String(rev.proposal_title ?? `Review ${i + 1}`)}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ═══════════════ NEW: Audit Dashboard ═══════════════ */}
      <div className="bg-surface-1 border border-surface-3 rounded-lg">
        <button
          onClick={() => setAuditOpen(!auditOpen)}
          className="w-full flex items-center justify-between p-4 text-left"
        >
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-gray-300">Decision Audit</h2>
            {totalDecisions > 0 && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-gray-600/40 text-gray-400">
                {totalDecisions.toLocaleString()} decisions (24h)
              </span>
            )}
          </div>
          <svg
            className={`w-4 h-4 text-gray-500 transition-transform ${auditOpen ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {auditOpen && (
          <div className="px-4 pb-4 space-y-4">
            {!auditSummary ? (
              <p className="text-xs text-gray-500">Loading audit data...</p>
            ) : (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {/* Decision Funnel */}
                  <div className="space-y-2">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Decision Funnel</h3>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <div className="h-6 bg-accent-blue/30 rounded" style={{ width: '100%' }} />
                        <span className="text-xs text-gray-300 font-mono shrink-0 w-20 text-right">
                          {totalDecisions.toLocaleString()}
                        </span>
                      </div>
                      <div className="text-[10px] text-gray-500 pl-1">Total decisions</div>

                      <div className="flex items-center gap-2">
                        <div
                          className="h-6 bg-accent-yellow/30 rounded"
                          style={{ width: `${totalDecisions ? (passedGates / totalDecisions) * 100 : 0}%`, minWidth: passedGates > 0 ? '8px' : '0' }}
                        />
                        <span className="text-xs text-gray-300 font-mono shrink-0 w-20 text-right">
                          {passedGates.toLocaleString()}
                        </span>
                      </div>
                      <div className="text-[10px] text-gray-500 pl-1">Passed gates</div>

                      <div className="flex items-center gap-2">
                        <div
                          className="h-6 bg-accent-green/30 rounded"
                          style={{ width: `${totalDecisions ? (totalTrades / totalDecisions) * 100 : 0}%`, minWidth: totalTrades > 0 ? '8px' : '0' }}
                        />
                        <span className="text-xs text-gray-300 font-mono shrink-0 w-20 text-right">
                          {totalTrades.toLocaleString()}
                        </span>
                      </div>
                      <div className="text-[10px] text-gray-500 pl-1">Executed trades</div>
                    </div>
                  </div>

                  {/* Gate Blocks Bar Chart */}
                  <div className="space-y-2">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Gate Blocks</h3>
                    {gateBlocks.length === 0 ? (
                      <p className="text-xs text-gray-500">No gate block data</p>
                    ) : (
                      <div className="space-y-1">
                        {gateBlocks.slice(0, 8).map((g) => (
                          <div key={g.gate} className="flex items-center gap-2">
                            <span className="text-[10px] text-gray-500 w-28 shrink-0 truncate text-right" title={g.gate}>
                              {g.gate.replace('gate_', '')}
                            </span>
                            <div className="flex-1 h-4 bg-surface-2 rounded overflow-hidden">
                              <div
                                className="h-full bg-accent-red/40 rounded"
                                style={{ width: `${(g.count / maxGateCount) * 100}%` }}
                              />
                            </div>
                            <span className="text-[10px] text-gray-400 font-mono w-10 text-right">{g.count}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Regime Distribution */}
                  <div className="space-y-2">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Regime Distribution</h3>
                    {regimeDist.length === 0 ? (
                      <p className="text-xs text-gray-500">No regime data</p>
                    ) : (
                      <>
                        {/* CSS donut chart */}
                        <div className="flex items-center gap-4">
                          <div
                            className="w-20 h-20 rounded-full shrink-0"
                            style={{
                              background: `conic-gradient(${
                                regimeDist.map((r, i) => {
                                  const startPct = regimeDist.slice(0, i).reduce((s, x) => s + x.count, 0) / totalRegime * 100;
                                  const endPct = startPct + (r.count / totalRegime) * 100;
                                  const color = REGIME_COLORS[r.label] || '#4b5563';
                                  return `${color} ${startPct}% ${endPct}%`;
                                }).join(', ')
                              })`,
                            }}
                          >
                            <div className="w-12 h-12 rounded-full bg-surface-1 m-4" />
                          </div>
                          <div className="space-y-0.5">
                            {regimeDist.map((r) => (
                              <div key={r.label} className="flex items-center gap-1.5 text-[10px]">
                                <span
                                  className="w-2 h-2 rounded-full shrink-0"
                                  style={{ backgroundColor: REGIME_COLORS[r.label] || '#4b5563' }}
                                />
                                <span className="text-gray-400">{r.label}</span>
                                <span className="text-gray-500 font-mono">{Math.round(r.count / totalRegime * 100)}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </>
                    )}

                    {/* ML Accuracy badge */}
                    {auditSummary && (auditSummary as Record<string, unknown>).ml_accuracy != null && (
                      <div className="mt-2 flex items-center gap-2">
                        <span className="text-[10px] text-gray-500 uppercase">ML Accuracy:</span>
                        <span className={`text-xs font-mono font-medium ${
                          ((auditSummary as Record<string, unknown>).ml_accuracy as number) > 0.52
                            ? 'text-accent-green'
                            : ((auditSummary as Record<string, unknown>).ml_accuracy as number) >= 0.50
                            ? 'text-accent-yellow'
                            : 'text-accent-red'
                        }`}>
                          {(((auditSummary as Record<string, unknown>).ml_accuracy as number) * 100).toFixed(1)}%
                        </span>
                        {(auditSummary as Record<string, unknown>).ml_evaluated != null && (
                          <span className="text-[10px] text-gray-600">
                            ({String((auditSummary as Record<string, unknown>).ml_evaluated)} evaluated)
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Recent Audit Trail */}
                <div>
                  <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Recent Audit Trail</h3>
                  {auditRecent.length === 0 ? (
                    <p className="text-xs text-gray-500">No audit rows yet</p>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-surface-3 text-gray-500">
                            <th className="text-left py-2 pr-2 font-medium">Time</th>
                            <th className="text-left py-2 pr-2 font-medium">Pair</th>
                            <th className="text-left py-2 pr-2 font-medium">Action</th>
                            <th className="text-left py-2 pr-2 font-medium">Blocked By</th>
                            <th className="text-left py-2 pr-2 font-medium">Regime</th>
                            <th className="text-right py-2 pr-2 font-medium">Confidence</th>
                            <th className="text-right py-2 font-medium">Signal</th>
                          </tr>
                        </thead>
                        <tbody>
                          {auditRecent.map((row, i) => (
                            <tr key={i} className="border-b border-surface-3 last:border-0">
                              <td className="py-1.5 pr-2 text-gray-500 font-mono">
                                {row.timestamp ? new Date(row.timestamp).toLocaleTimeString() : '--'}
                              </td>
                              <td className="py-1.5 pr-2 text-gray-300">{row.pair ?? '--'}</td>
                              <td className={`py-1.5 pr-2 font-medium ${
                                row.action === 'BUY' ? 'text-accent-green'
                                  : row.action === 'SELL' ? 'text-accent-red'
                                  : 'text-gray-500'
                              }`}>
                                {row.action ?? '--'}
                              </td>
                              <td className="py-1.5 pr-2 text-gray-500">
                                {row.blocked_by_gate
                                  ? <span className="text-accent-red">{String(row.blocked_by_gate).replace('gate_', '')}</span>
                                  : <span className="text-accent-green">--</span>
                                }
                              </td>
                              <td className="py-1.5 pr-2 text-gray-400">{row.regime_label ?? '--'}</td>
                              <td className="py-1.5 pr-2 text-right font-mono text-gray-400">
                                {row.confidence != null ? (row.confidence as number).toFixed(2) : '--'}
                              </td>
                              <td className="py-1.5 text-right font-mono text-gray-400">
                                {row.weighted_signal != null ? (row.weighted_signal as number).toFixed(4) : '--'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </PageShell>
  );
}
