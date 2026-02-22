import { useEffect, useState, useCallback, useRef } from 'react';
import { api } from '../../api';
import type { BacktestConfig, BacktestProgress, BacktestSummary } from '../../types';
import { formatCurrency } from '../../utils/formatters';

const ALL_PAIRS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'LINK-USD', 'AVAX-USD', 'DOGE-USD'];

const CURRENT_PRESET: BacktestConfig = {
  pairs: [...ALL_PAIRS],
  cost_bps: 0.0065,
  lookahead: 6,
  new_denom: 0.02,
  new_buy_thresh: 0.015,
  new_sell_thresh: -0.015,
  new_conf_floor: 0.48,
  new_signal_scale: 10.0,
  new_exit_bars: 6,
  new_pos_min: 50,
  new_pos_max: 300,
  new_pos_base: 100,
  old_denom: 0.05,
  old_buy_thresh: 0.01,
  old_sell_thresh: -0.01,
  old_conf_floor: 0.505,
  old_signal_scale: 1.0,
  old_exit_bars: 1,
  old_pos_usd: 75,
};

const LEGACY_PRESET: BacktestConfig = {
  ...CURRENT_PRESET,
  new_denom: 0.05,
  new_buy_thresh: 0.01,
  new_sell_thresh: -0.01,
  new_conf_floor: 0.505,
  new_signal_scale: 1.0,
  new_exit_bars: 1,
  new_pos_min: 75,
  new_pos_max: 75,
  new_pos_base: 75,
};

interface Props {
  onComplete?: () => void;
}

export default function BacktestRunnerPanel({ onComplete }: Props) {
  const [config, setConfig] = useState<BacktestConfig>({ ...CURRENT_PRESET });
  const [state, setState] = useState<'idle' | 'running' | 'complete' | 'error'>('idle');
  const [progress, setProgress] = useState<BacktestProgress | null>(null);
  const [summary, setSummary] = useState<BacktestSummary | null>(null);
  const [csvPath, setCsvPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const mountedRef = useRef(true);

  // On mount: poll status once to recover mid-run state
  useEffect(() => {
    mountedRef.current = true;
    api.backtestStatus().then((s) => {
      if (!mountedRef.current) return;
      if (s.state === 'running') {
        setState('running');
        if (s.progress) setProgress(s.progress as BacktestProgress);
        if (s.config) setConfig(s.config as BacktestConfig);
      } else if (s.state === 'complete') {
        setState('complete');
        if (s.summary) setSummary(s.summary as BacktestSummary);
        if (s.csv_path) setCsvPath(s.csv_path);
      } else if (s.state === 'error') {
        setState('error');
        if (s.error) setError(s.error);
      }
    }).catch(() => {});
    return () => { mountedRef.current = false; };
  }, []);

  // Listen for WS progress events
  useEffect(() => {
    const handler = (e: Event) => {
      const data = (e as CustomEvent).detail as BacktestProgress;
      if (!mountedRef.current) return;
      if (data.state === 'running') {
        setState('running');
        setProgress(data);
      } else if (data.state === 'complete') {
        setState('complete');
        if (data.summary) setSummary(data.summary);
        if (data.csv_path) setCsvPath(data.csv_path);
        onComplete?.();
      } else if (data.state === 'error') {
        setState('error');
        setError(data.error ?? 'Unknown error');
      }
    };
    window.addEventListener('backtest-progress', handler);
    return () => window.removeEventListener('backtest-progress', handler);
  }, [onComplete]);

  const handleStart = useCallback(async () => {
    try {
      setState('running');
      setProgress(null);
      setSummary(null);
      setCsvPath(null);
      setError(null);
      await api.startBacktest(config);
    } catch (e: any) {
      setState('error');
      setError(e?.message || 'Failed to start backtest');
    }
  }, [config]);

  const togglePair = useCallback((pair: string) => {
    setConfig(prev => {
      const pairs = prev.pairs.includes(pair)
        ? prev.pairs.filter(p => p !== pair)
        : [...prev.pairs, pair];
      return { ...prev, pairs };
    });
  }, []);

  const setPreset = useCallback((preset: BacktestConfig) => {
    setConfig({ ...preset });
  }, []);

  const updateField = useCallback((field: keyof BacktestConfig, value: number) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  }, []);

  const pct = progress?.pct ?? 0;
  const isRunning = state === 'running';

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Backtest Runner
        </h2>
        {state === 'complete' && csvPath && (
          <button
            onClick={() => window.open(api.backtestDownloadUrl(), '_blank')}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-cyan-500/10 text-cyan-400 rounded-lg text-xs hover:bg-cyan-500/20 transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download CSV
          </button>
        )}
      </div>

      {/* Presets */}
      <div className="flex gap-2">
        <button
          onClick={() => setPreset(CURRENT_PRESET)}
          disabled={isRunning}
          className="px-3 py-1.5 bg-blue-500/10 text-blue-400 rounded-lg text-xs hover:bg-blue-500/20 transition-colors disabled:opacity-40"
        >
          Current Pipeline
        </button>
        <button
          onClick={() => setPreset(LEGACY_PRESET)}
          disabled={isRunning}
          className="px-3 py-1.5 bg-gray-500/10 text-gray-400 rounded-lg text-xs hover:bg-gray-500/20 transition-colors disabled:opacity-40"
        >
          Legacy Pipeline
        </button>
      </div>

      {/* Pair toggles */}
      <div className="flex flex-wrap gap-2">
        {ALL_PAIRS.map(pair => (
          <button
            key={pair}
            onClick={() => togglePair(pair)}
            disabled={isRunning}
            className={`px-2.5 py-1 rounded text-xs font-mono transition-colors disabled:opacity-50 ${
              config.pairs.includes(pair)
                ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                : 'bg-surface-2 text-gray-500 border border-surface-3'
            }`}
          >
            {pair.replace('-USD', '')}
          </button>
        ))}
      </div>

      {/* Cost BPS input */}
      <div className="flex items-center gap-3">
        <label className="text-xs text-gray-500 w-24">Cost (bps):</label>
        <input
          type="number"
          step="0.001"
          min="0"
          max="0.05"
          value={config.cost_bps}
          onChange={e => updateField('cost_bps', parseFloat(e.target.value) || 0)}
          disabled={isRunning}
          className="w-24 bg-surface-2 border border-surface-3 rounded px-2 py-1 text-xs text-gray-300 font-mono disabled:opacity-50"
        />
        <span className="text-xs text-gray-600">({(config.cost_bps * 100).toFixed(2)}% round-trip)</span>
      </div>

      {/* Advanced params */}
      <details open={showAdvanced} onToggle={(e) => setShowAdvanced((e.target as HTMLDetailsElement).open)}>
        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400 select-none">
          Advanced Parameters
        </summary>
        <div className="mt-3 grid grid-cols-2 gap-4">
          {/* FIXED column */}
          <div className="space-y-2">
            <h4 className="text-[10px] font-semibold text-blue-400 uppercase tracking-wider">FIXED Pipeline</h4>
            {([
              ['new_denom', 'Denominator', 0.001],
              ['new_buy_thresh', 'Buy Thresh', 0.001],
              ['new_sell_thresh', 'Sell Thresh', 0.001],
              ['new_conf_floor', 'Conf Floor', 0.01],
              ['new_signal_scale', 'Signal Scale', 0.5],
              ['new_exit_bars', 'Exit Bars', 1],
              ['new_pos_min', 'Pos Min $', 5],
              ['new_pos_max', 'Pos Max $', 10],
              ['new_pos_base', 'Pos Base $', 5],
            ] as [keyof BacktestConfig, string, number][]).map(([key, label, step]) => (
              <div key={key} className="flex items-center gap-2">
                <label className="text-[10px] text-gray-500 w-20 shrink-0">{label}</label>
                <input
                  type="number"
                  step={step}
                  value={config[key] as number}
                  onChange={e => updateField(key, parseFloat(e.target.value) || 0)}
                  disabled={isRunning}
                  className="flex-1 min-w-0 bg-surface-2 border border-surface-3 rounded px-1.5 py-0.5 text-[10px] text-gray-300 font-mono disabled:opacity-50"
                />
              </div>
            ))}
          </div>

          {/* OLD column */}
          <div className="space-y-2">
            <h4 className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">OLD Pipeline</h4>
            {([
              ['old_denom', 'Denominator', 0.001],
              ['old_buy_thresh', 'Buy Thresh', 0.001],
              ['old_sell_thresh', 'Sell Thresh', 0.001],
              ['old_conf_floor', 'Conf Floor', 0.01],
              ['old_signal_scale', 'Signal Scale', 0.5],
              ['old_exit_bars', 'Exit Bars', 1],
              ['old_pos_usd', 'Pos USD $', 5],
            ] as [keyof BacktestConfig, string, number][]).map(([key, label, step]) => (
              <div key={key} className="flex items-center gap-2">
                <label className="text-[10px] text-gray-500 w-20 shrink-0">{label}</label>
                <input
                  type="number"
                  step={step}
                  value={config[key] as number}
                  onChange={e => updateField(key, parseFloat(e.target.value) || 0)}
                  disabled={isRunning}
                  className="flex-1 min-w-0 bg-surface-2 border border-surface-3 rounded px-1.5 py-0.5 text-[10px] text-gray-300 font-mono disabled:opacity-50"
                />
              </div>
            ))}
          </div>
        </div>
      </details>

      {/* Run button */}
      <button
        onClick={handleStart}
        disabled={isRunning || config.pairs.length === 0}
        className={`w-full py-2.5 rounded-lg text-sm font-medium transition-colors ${
          isRunning
            ? 'bg-cyan-500/10 text-cyan-400 cursor-not-allowed'
            : 'bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30'
        } disabled:opacity-50`}
      >
        {isRunning ? 'Backtest Running...' : 'Run Backtest'}
      </button>

      {/* Progress bar */}
      {isRunning && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-400">
              {progress?.pair ? `Processing ${progress.pair}` : 'Starting...'}{' '}
              {progress?.pair_idx != null && progress?.total_pairs != null && (
                <span className="text-gray-600">({progress.pair_idx}/{progress.total_pairs})</span>
              )}
            </span>
            <span className="text-cyan-400 font-mono">{pct.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-surface-3 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(pct, 100)}%` }}
            />
          </div>
          {progress?.new_pnl != null && progress?.old_pnl != null && (
            <div className="flex justify-between text-[10px] font-mono">
              <span className="text-blue-400">FIXED: {formatCurrency(progress.new_pnl)}</span>
              <span className="text-gray-500">OLD: {formatCurrency(progress.old_pnl)}</span>
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {state === 'error' && error && (
        <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-3">
          <p className="text-xs text-red-400 font-mono whitespace-pre-wrap max-h-32 overflow-y-auto">{error}</p>
        </div>
      )}

      {/* Results summary */}
      {state === 'complete' && summary && (
        <ResultsSummary summary={summary} />
      )}
    </div>
  );
}

function ResultsSummary({ summary }: { summary: BacktestSummary }) {
  const n = summary.new;
  const o = summary.old;

  const rows: [string, string, string, string][] = [
    ['Trades', `${n.n_trades}`, `${o.n_trades}`, `${n.n_trades - o.n_trades > 0 ? '+' : ''}${n.n_trades - o.n_trades}`],
    ['Win Rate', `${n.win_rate.toFixed(1)}%`, `${o.win_rate.toFixed(1)}%`, `${(n.win_rate - o.win_rate) >= 0 ? '+' : ''}${(n.win_rate - o.win_rate).toFixed(1)}pp`],
    ['Total P&L', formatCurrency(n.total_pnl), formatCurrency(o.total_pnl), formatCurrency(n.total_pnl - o.total_pnl)],
    ['Avg P&L', formatCurrency(n.avg_pnl), formatCurrency(o.avg_pnl), formatCurrency(n.avg_pnl - o.avg_pnl)],
    ['Sharpe', n.sharpe.toFixed(2), o.sharpe.toFixed(2), `${(n.sharpe - o.sharpe) >= 0 ? '+' : ''}${(n.sharpe - o.sharpe).toFixed(2)}`],
    ['Max DD', formatCurrency(n.max_drawdown), formatCurrency(o.max_drawdown), formatCurrency(n.max_drawdown - o.max_drawdown)],
    ['Avg Pos', formatCurrency(n.avg_position_usd), formatCurrency(o.avg_position_usd), ''],
  ];

  const pnlDelta = n.total_pnl - o.total_pnl;

  return (
    <div className="space-y-3">
      {/* Verdict */}
      <div className={`text-center py-2 rounded-lg text-xs font-medium ${
        pnlDelta > 0
          ? 'bg-green-500/5 border border-green-500/20 text-green-400'
          : pnlDelta < 0
          ? 'bg-red-500/5 border border-red-500/20 text-red-400'
          : 'bg-gray-500/5 border border-gray-500/20 text-gray-400'
      }`}>
        FIXED pipeline {pnlDelta > 0 ? 'outperformed' : pnlDelta < 0 ? 'underperformed' : 'matched'} OLD by {formatCurrency(Math.abs(pnlDelta))}
        <span className="text-gray-600 ml-2">({summary.total_inferences.toLocaleString()} inferences in {summary.elapsed.toFixed(1)}s)</span>
      </div>

      {/* Comparison table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-gray-500 border-b border-surface-3">
              <th className="text-left py-1.5 px-2">Metric</th>
              <th className="text-right py-1.5 px-2 text-blue-400">FIXED</th>
              <th className="text-right py-1.5 px-2 text-gray-500">OLD</th>
              <th className="text-right py-1.5 px-2">Delta</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, fixed, old, delta]) => (
              <tr key={label} className="border-b border-surface-3/30">
                <td className="py-1.5 px-2 text-gray-400">{label}</td>
                <td className="py-1.5 px-2 text-right text-blue-300">{fixed}</td>
                <td className="py-1.5 px-2 text-right text-gray-500">{old}</td>
                <td className="py-1.5 px-2 text-right text-gray-300">{delta}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-pair breakdown */}
      {summary.per_pair && Object.keys(summary.per_pair).length > 0 && (
        <details>
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400 select-none">
            Per-Pair Breakdown
          </summary>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-[10px] font-mono">
              <thead>
                <tr className="text-gray-600 border-b border-surface-3">
                  <th className="text-left py-1 px-1.5">Pair</th>
                  <th className="text-right py-1 px-1.5">FIXED P&L</th>
                  <th className="text-right py-1 px-1.5">OLD P&L</th>
                  <th className="text-right py-1 px-1.5">Delta</th>
                  <th className="text-right py-1 px-1.5">FIXED WR</th>
                  <th className="text-right py-1 px-1.5">OLD WR</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(summary.per_pair).sort().map(([pair, { new: pn, old: po }]) => {
                  const d = pn.total_pnl - po.total_pnl;
                  return (
                    <tr key={pair} className="border-b border-surface-3/20">
                      <td className="py-1 px-1.5 text-gray-400">{pair}</td>
                      <td className={`py-1 px-1.5 text-right ${pn.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>{formatCurrency(pn.total_pnl)}</td>
                      <td className={`py-1 px-1.5 text-right ${po.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>{formatCurrency(po.total_pnl)}</td>
                      <td className={`py-1 px-1.5 text-right ${d >= 0 ? 'text-green-400' : 'text-red-400'}`}>{formatCurrency(d)}</td>
                      <td className="py-1 px-1.5 text-right text-gray-400">{pn.win_rate.toFixed(1)}%</td>
                      <td className="py-1 px-1.5 text-right text-gray-500">{po.win_rate.toFixed(1)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}
