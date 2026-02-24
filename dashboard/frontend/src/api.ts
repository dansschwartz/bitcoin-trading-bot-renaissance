/**
 * REST API fetch helpers.
 */

const BASE = '';  // Same origin in production, proxied in dev

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail || `API ${res.status}: ${path}`);
  }
  return res.json();
}

// ─── System ──────────────────────────────────────────────────────────────
import type {
  SystemStatus, SystemConfig, Decision, Trade, Position, EquityPoint,
  PnLSummary, RegimePerformance, CalendarPnL, HourlyPnL, MLPrediction,
  EnsembleStatus, RegimeStatus, VAEPoint, RiskMetrics, Exposure,
  RiskGatewayEntry, TimeRange, BacktestRun, BacktestComparison,
  ClosedPosition, PositionSummary, ArbStatus, ArbTrade, ArbSignal, ArbSummary,
  ArbWallet, BacktestConfig, BacktestProgress,
} from './types';

export const api = {
  // System
  status: () => get<SystemStatus>('/api/system/status'),
  config: () => get<SystemConfig>('/api/system/config'),
  health: () => get<{ status: string }>('/api/health'),

  // Decisions
  recentDecisions: (limit = 100) => get<Decision[]>(`/api/decisions/recent?limit=${limit}`),
  decision: (id: number) => get<Decision>(`/api/decisions/${id}`),
  signalWeights: () => get<{ weights: Record<string, number> }>('/api/signals/weights'),

  // Trades & Positions
  openPositions: () => get<Position[]>('/api/positions/open'),
  closedTrades: (limit = 50, offset = 0) =>
    get<Trade[]>(`/api/trades/closed?limit=${limit}&offset=${offset}`),
  tradeLifecycle: (id: number) => get<{ trade: Trade; context_decisions: Decision[] }>(
    `/api/trades/${id}/lifecycle`,
  ),

  // Analytics
  equity: (range: TimeRange = '1D') => get<EquityPoint[]>(`/api/analytics/equity?range=${range}`),
  pnl: (range: TimeRange = '1D') => get<PnLSummary>(`/api/analytics/pnl?range=${range}`),
  byRegime: () => get<RegimePerformance[]>('/api/analytics/by-regime'),
  byExecution: () => get<{ algo_used: string; count: number; avg_slippage: number }[]>(
    '/api/analytics/by-execution',
  ),
  distribution: () => get<{ trade_pnl: number }[]>('/api/analytics/distribution'),
  calendar: () => get<CalendarPnL[]>('/api/analytics/calendar'),
  hourly: () => get<HourlyPnL[]>('/api/analytics/hourly'),
  benchmark: (range: TimeRange = '1D', productId = 'BTC-USD') =>
    get<{ timestamp: string; benchmark_equity: number }[]>(
      `/api/analytics/benchmark?range=${range}&product_id=${productId}`,
    ),
  signalAttribution: (hours = 24) =>
    get<Record<string, unknown>>(`/api/analytics/signal-attribution?hours=${hours}`),
  modelAccuracy: (hours = 24) =>
    get<Record<string, unknown>>(`/api/analytics/model-accuracy?hours=${hours}`),

  // Brain
  ensemble: () => get<EnsembleStatus>('/api/brain/ensemble'),
  predictionHistory: (hours = 24) => get<MLPrediction[]>(`/api/brain/predictions/history?hours=${hours}`),
  regime: () => get<RegimeStatus>('/api/brain/regime'),
  confluence: () => get<Record<string, unknown>>('/api/brain/confluence'),
  vae: () => get<VAEPoint[]>('/api/brain/vae'),

  // Risk
  exposure: () => get<Exposure>('/api/risk/exposure'),
  riskMetrics: () => get<RiskMetrics>('/api/risk/metrics'),
  gatewayLog: (limit = 100) => get<RiskGatewayEntry[]>(`/api/risk/gateway/log?limit=${limit}`),
  alerts: () => get<{ alerts: Record<string, unknown>[] }>('/api/risk/alerts'),

  // Backtest
  backtestRuns: () => get<BacktestRun[]>('/api/backtest/runs'),
  backtestResult: (id: number) => get<BacktestRun>(`/api/backtest/runs/${id}`),
  backtestCompare: (id: number) => get<BacktestComparison>(`/api/backtest/compare/${id}`),
  startBacktest: (config: BacktestConfig) => post<{ status: string }>('/api/backtest/start', config),
  backtestStatus: () => get<BacktestProgress>('/api/backtest/status'),
  backtestDownloadUrl: () => `${BASE}/api/backtest/download`,

  // Positions (round-trip P&L)
  closedPositions: (limit = 50, offset = 0) =>
    get<ClosedPosition[]>(`/api/positions/closed?limit=${limit}&offset=${offset}`),
  positionSummary: () => get<PositionSummary>('/api/positions/summary'),

  // Arbitrage
  arbStatus: () => get<ArbStatus>('/api/arbitrage/status'),
  arbTrades: (limit = 50, offset = 0, strategy?: string) =>
    get<ArbTrade[]>(`/api/arbitrage/trades?limit=${limit}&offset=${offset}${strategy ? `&strategy=${strategy}` : ''}`),
  arbSignals: (limit = 100, strategy?: string) =>
    get<ArbSignal[]>(`/api/arbitrage/signals?limit=${limit}${strategy ? `&strategy=${strategy}` : ''}`),
  arbSummary: () => get<ArbSummary>('/api/arbitrage/summary'),
  arbWallet: () => get<ArbWallet>('/api/arbitrage/wallet'),

  // Breakout Scanner
  breakoutSummary: () => get<Record<string, unknown>>('/api/breakout/summary'),
  breakoutSignals: (limit = 30) => get<Record<string, unknown>>(`/api/breakout/signals?limit=${limit}`),
  breakoutHistory: (hours = 24, limit = 100) =>
    get<Record<string, unknown>>(`/api/breakout/history?hours=${hours}&limit=${limit}`),
  breakoutHeatmap: () => get<Record<string, unknown>>('/api/breakout/heatmap'),

  // Polymarket
  polymarketSummary: () => get<Record<string, unknown>>('/api/polymarket/summary'),
  polymarketEdges: (minEdge = 0, limit = 50) =>
    get<Record<string, unknown>>(`/api/polymarket/edges?min_edge=${minEdge}&limit=${limit}`),
  polymarketMarkets: (marketType?: string, asset?: string, limit = 100) => {
    let path = `/api/polymarket/markets?limit=${limit}`;
    if (marketType) path += `&market_type=${marketType}`;
    if (asset) path += `&asset=${asset}`;
    return get<Record<string, unknown>>(path);
  },
  polymarketSignal: () => get<Record<string, unknown>>('/api/polymarket/signal'),
  polymarketHistory: (hours = 24, asset?: string) =>
    get<Record<string, unknown>>(`/api/polymarket/history?hours=${hours}${asset ? `&asset=${asset}` : ''}`),
  polymarketStats: () => get<Record<string, unknown>>('/api/polymarket/stats'),
  polymarketPositions: (status?: string, limit = 100) => {
    let path = `/api/polymarket/positions?limit=${limit}`;
    if (status) path += `&status=${status}`;
    return get<Record<string, unknown>>(path);
  },
  polymarketPnl: () => get<Record<string, unknown>>('/api/polymarket/pnl'),
  polymarketExecutor: () => get<Record<string, unknown>>('/api/polymarket/executor'),
  polymarketInstruments: () => get<Record<string, unknown>>('/api/polymarket/instruments'),
  polymarketLifecycle: (limit = 50) => get<Record<string, unknown>>(`/api/polymarket/lifecycle?limit=${limit}`),
  polymarketLifecycleStats: () => get<Record<string, unknown>>('/api/polymarket/lifecycle/stats'),

  // Devil Tracker
  devilSummary: (hours = 24) => get<Record<string, unknown>>(`/api/devil/summary?hours=${hours}`),

  // Agents (Doc 15)
  agentStatuses: () => get<Record<string, unknown>[]>('/api/agents/status'),
  agentEvents: (limit = 100, agent?: string) =>
    get<Record<string, unknown>[]>(`/api/agents/events?limit=${limit}${agent ? `&agent=${agent}` : ''}`),
  agentProposals: (status?: string, limit = 50) =>
    get<Record<string, unknown>[]>(`/api/agents/proposals?limit=${limit}${status ? `&status=${status}` : ''}`),
  agentImprovements: (limit = 50) =>
    get<Record<string, unknown>[]>(`/api/agents/improvements?limit=${limit}`),
  agentLatestReport: () => get<Record<string, unknown>>('/api/agents/reports/latest'),
  agentModels: () => get<Record<string, unknown>[]>('/api/agents/models'),
};
