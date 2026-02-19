// ─── System ──────────────────────────────────────────────────────────────

export interface SystemStatus {
  status: string;
  uptime_seconds: number;
  cycle_count: number;
  trade_count: number;
  open_position_count: number;
  paper_trading: boolean;
  product_ids: string[];
  latest_prices: Record<string, PriceSnapshot>;
  ws_clients: number;
  timestamp: string;
}

export interface PriceSnapshot {
  price: number;
  bid: number;
  ask: number;
  timestamp: string;
  change_pct?: number;
}

export interface SystemConfig {
  flags: Record<string, boolean>;
  signal_weights: Record<string, number>;
  dashboard: DashboardDisplayConfig;
  product_ids: string[];
  paper_trading: boolean;
}

export interface DashboardDisplayConfig {
  host: string;
  port: number;
  refresh_interval_ms: number;
  dark_mode: boolean;
  alerts: Record<string, number>;
  display: Record<string, number>;
}

// ─── Decisions ───────────────────────────────────────────────────────────

export interface Decision {
  id: number;
  timestamp: string;
  product_id: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  position_size: number;
  weighted_signal: number;
  reasoning: Record<string, unknown>;
  feature_vector?: string;
  vae_loss?: number;
  hmm_regime?: string;
}

// ─── Trades & Positions ─────────────────────────────────────────────────

export interface Trade {
  id: number;
  timestamp: string;
  product_id: string;
  side: string;
  size: number;
  price: number;
  status: string;
  algo_used?: string;
  slippage?: number;
  execution_time?: number;
}

export interface Position {
  position_id: string;
  product_id: string;
  side: string;
  size: number;
  entry_price: number;
  stop_loss_price?: number;
  take_profit_price?: number;
  opened_at: string;
  status: string;
  unrealized_pnl?: number;
  current_price?: number;
}

// ─── Analytics ───────────────────────────────────────────────────────────

export interface EquityPoint {
  timestamp: string;
  side: string;
  size: number;
  price: number;
  pnl_delta: number;
  cumulative_pnl: number;
  equity?: number;
}

export interface PnLSummary {
  total_trades: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
  avg_slippage: number;
  win_rate: number;
  total_round_trips: number;
  winning_round_trips: number;
  initial_capital: number;
  current_equity: number;
}

export interface RegimePerformance {
  regime: string;
  action: string;
  count: number;
  avg_confidence: number;
  avg_signal: number;
}

export interface CalendarPnL {
  date: string;
  daily_pnl: number;
  trade_count: number;
}

export interface HourlyPnL {
  hour: number;
  avg_pnl: number;
  trade_count: number;
}

// ─── Brain ───────────────────────────────────────────────────────────────

export interface MLPrediction {
  id: number;
  timestamp: string;
  product_id: string;
  model_name: string;
  prediction: number;
}

export interface EnsembleStatus {
  models: Record<string, MLPrediction>;
  model_count: number;
}

export interface RegimeStatus {
  current: Decision | null;
  history: Decision[];
}

export interface VAEPoint {
  id: number;
  timestamp: string;
  vae_loss: number;
}

// ─── Risk ────────────────────────────────────────────────────────────────

export interface RiskMetrics {
  max_drawdown: number;
  cumulative_pnl: number;
  peak_equity: number;
  current_equity: number;
  initial_capital: number;
  sharpe_ratio: number;
  max_consecutive_losses: number;
  total_trading_days: number;
  unrealized_pnl: number;
}

export interface AssetExposure {
  count: number;
  long_size: number;
  short_size: number;
  long_value: number;
  short_value: number;
  net_size: number;
  net_value: number;
  total_size: number;
  total_value: number;
}

export interface Exposure {
  long_exposure: number;
  short_exposure: number;
  net_exposure: number;
  gross_exposure: number;
  position_count: number;
  positions_by_asset: Record<string, AssetExposure>;
}

export interface RiskGatewayEntry {
  id: number;
  timestamp: string;
  product_id: string;
  action: string;
  confidence: number;
  vae_loss: number;
  hmm_regime: string;
  gateway_verdict?: string;
}

// ─── WebSocket ───────────────────────────────────────────────────────────

export interface WSMessage {
  channel: string;
  data: Record<string, unknown>;
  ts: string;
}

// ─── Backtest ───────────────────────────────────────────────────────────

export interface BacktestRun {
  id: number;
  timestamp: string;
  config: Record<string, unknown>;
  total_trades: number;
  realized_pnl: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
}

export interface BacktestComparison {
  backtest: BacktestRun;
  live: {
    risk_metrics: RiskMetrics;
    pnl_summary: PnLSummary;
  };
}

// ─── Time Range ──────────────────────────────────────────────────────────

export type TimeRange = '1H' | '4H' | '1D' | '1W' | '1M' | 'ALL';

// ─── Closed Positions (round-trip P&L) ──────────────────────────────────

export interface ClosedPosition {
  position_id: string;
  product_id: string;
  side: string;
  size: number;
  entry_price: number;
  close_price: number | null;
  opened_at: string;
  closed_at: string | null;
  realized_pnl: number | null;
  exit_reason: string | null;
  hold_duration_seconds: number | null;
}

export interface PositionSummary {
  total_closed: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_realized_pnl: number;
  avg_win: number;
  avg_loss: number;
  largest_win: number;
  largest_loss: number;
  avg_hold_seconds: number;
}

// ─── Arbitrage ──────────────────────────────────────────────────────────

export interface ArbStatus {
  running: boolean;
  uptime_seconds: number;
  book_status?: Record<string, unknown>;
  cross_exchange?: Record<string, unknown>;
  executor?: Record<string, unknown>;
  funding?: Record<string, unknown>;
  triangular?: Record<string, unknown>;
  risk?: Record<string, unknown>;
  tracker_summary?: Record<string, unknown>;
  db_summary?: Record<string, unknown>;
}

export interface ArbTrade {
  id: number;
  trade_id: string;
  strategy: string;
  symbol: string;
  buy_exchange: string;
  sell_exchange: string;
  status: string;
  buy_price: number;
  sell_price: number;
  quantity: number;
  gross_spread_bps: number;
  net_spread_bps: number;
  expected_profit_usd: number;
  actual_profit_usd: number;
  timestamp: string;
}

export interface ArbSignal {
  id: number;
  signal_id: string;
  strategy: string;
  symbol: string;
  gross_spread_bps: number;
  net_spread_bps: number;
  approved: number;
  executed: number;
  timestamp: string;
}

export interface ArbSummary {
  total_trades: number;
  filled_trades: number;
  total_profit_usd: number;
  wins: number;
  losses: number;
  win_rate: number;
  signals_total: number;
  signals_approved: number;
  daily_pnl_usd: number;
  by_strategy: { strategy: string; trades: number; fills: number; profit_usd: number }[];
}

export interface ArbWallet {
  initial_balance: number;
  current_balance: number;
  total_realized_pnl: number;
  return_pct: number;
}
