import { useEffect, useState } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import { api } from '../../api';
import { formatCurrency, formatUptime } from '../../utils/formatters';
import type { Exposure, RegimeStatus } from '../../types';

const INITIAL_CAPITAL = 410_000;

type Health = 'green' | 'yellow' | 'red';

interface HealthIndicator {
  label: string;
  value: string;
  health: Health;
}

interface AlertData {
  type: string;
  severity: string;
  message: string;
}

function HealthDot({ health }: { health: Health }) {
  const color = health === 'green'
    ? 'bg-accent-green' : health === 'yellow'
    ? 'bg-yellow-400' : 'bg-accent-red';
  return <span className={`inline-block w-1.5 h-1.5 rounded-full ${color}`} />;
}

function computeRegimeHealth(regime: RegimeStatus | null): HealthIndicator {
  if (!regime?.current) return { label: 'Regime', value: 'No data', health: 'red' };
  const current = regime.current as unknown as Record<string, unknown>;
  const name = (current.hmm_regime as string) ?? 'unknown';
  const confidence = (current.confidence as number) ?? 0;
  const classifier = (current.classifier as string) ?? 'none';

  const tag = classifier === 'hmm' ? '[HMM]' : classifier === 'bootstrap' ? '[Boot]' : '';

  if (name === 'unknown') {
    return {
      label: 'Regime',
      value: classifier === 'none' ? 'Warming up' : `UNKNOWN ${tag}`,
      health: classifier === 'none' ? 'yellow' : 'red',
    };
  }

  const displayName = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  const confPct = `${(confidence * 100).toFixed(0)}%`;
  return {
    label: 'Regime',
    value: `${displayName} ${confPct} ${tag}`,
    health: confidence > 0.6 ? 'green' : 'yellow',
  };
}

function computeRiskHealth(alerts: AlertData[]): HealthIndicator {
  const critCount = alerts.filter(a => a.severity === 'CRITICAL').length;
  const warnCount = alerts.filter(a => a.severity === 'WARNING').length;

  if (critCount > 0) {
    return { label: 'Risk', value: `${critCount} CRITICAL`, health: 'red' };
  }
  if (warnCount > 0) {
    return { label: 'Risk', value: `${warnCount} warning${warnCount > 1 ? 's' : ''}`, health: 'yellow' };
  }
  return { label: 'Risk', value: 'Clear', health: 'green' };
}

function computeDataHealth(prices: Record<string, { timestamp?: string }> | undefined): HealthIndicator {
  if (!prices || Object.keys(prices).length === 0) {
    return { label: 'Data', value: 'No feed', health: 'red' };
  }

  const now = Date.now();
  let maxLagMs = 0;
  let staleCount = 0;
  for (const snap of Object.values(prices)) {
    if (!snap.timestamp) { staleCount++; continue; }
    const lag = now - new Date(snap.timestamp).getTime();
    if (lag > maxLagMs) maxLagMs = lag;
    if (lag > 30_000) staleCount++;
  }

  const lagSec = (maxLagMs / 1000).toFixed(1);
  if (staleCount > 0) {
    return { label: 'Data', value: `${staleCount} stale (${lagSec}s)`, health: 'red' };
  }
  if (maxLagMs > 10_000) {
    return { label: 'Data', value: `Live (${lagSec}s lag)`, health: 'yellow' };
  }
  return { label: 'Data', value: `Live (${lagSec}s)`, health: 'green' };
}

function computeExposureHealth(exposure: Exposure | null): HealthIndicator {
  if (!exposure) return { label: 'Exposure', value: 'No data', health: 'yellow' };

  const grossPct = (exposure.gross_exposure / INITIAL_CAPITAL) * 100;
  const display = `${grossPct.toFixed(0)}% of capital`;

  if (grossPct > 80) return { label: 'Exposure', value: display, health: 'red' };
  if (grossPct > 50) return { label: 'Exposure', value: display, health: 'yellow' };
  return { label: 'Exposure', value: display, health: 'green' };
}

function computeMLHealth(modelCount: number): HealthIndicator {
  if (modelCount >= 4) return { label: 'ML Models', value: `${modelCount}/${modelCount} active`, health: 'green' };
  if (modelCount > 0) return { label: 'ML Models', value: `${modelCount} active`, health: 'yellow' };
  return { label: 'ML Models', value: 'None', health: 'red' };
}

function computePnLHealth(totalPnl: number): HealthIndicator {
  const pctChange = (totalPnl / INITIAL_CAPITAL) * 100;
  const display = formatCurrency(totalPnl);

  if (totalPnl >= 0) return { label: 'Daily P&L', value: `+${display}`, health: 'green' };
  if (pctChange > -2) return { label: 'Daily P&L', value: display, health: 'yellow' };
  return { label: 'Daily P&L', value: display, health: 'red' };
}

export default function SystemHealthBar() {
  const { state } = useDashboard();
  const { status, config, pnl } = state;

  const [regime, setRegime] = useState<RegimeStatus | null>(null);
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [exposure, setExposure] = useState<Exposure | null>(null);
  const [modelCount, setModelCount] = useState(0);

  useEffect(() => {
    const fetchHealth = () => {
      api.regime().then(setRegime).catch(() => {});
      api.alerts().then(d => setAlerts((d.alerts ?? []) as unknown as AlertData[])).catch(() => {});
      api.exposure().then(setExposure).catch(() => {});
      api.ensemble().then(d => setModelCount(d.model_count ?? Object.keys(d.models ?? {}).length)).catch(() => {});
    };
    fetchHealth();
    const id = setInterval(fetchHealth, 15_000);
    return () => clearInterval(id);
  }, []);

  const flags = config?.flags ?? {};
  const uptime = status?.uptime_seconds ?? 0;
  const cycles = status?.cycle_count ?? 0;
  const cyclesPerMin = uptime > 60 ? (cycles / (uptime / 60)).toFixed(1) : '--';
  const isPaper = status?.paper_trading ?? true;

  // Compute health indicators
  const indicators: HealthIndicator[] = [
    computeRegimeHealth(regime),
    computeRiskHealth(alerts),
    computeDataHealth(status?.latest_prices),
    computeExposureHealth(exposure),
    computeMLHealth(modelCount),
    computePnLHealth(pnl?.total_pnl ?? pnl?.realized_pnl ?? 0),
  ];

  const MODULE_LABELS: Record<string, string> = {
    enable_ml: 'ML', enable_hmm: 'HMM', enable_vae: 'VAE',
    enable_adaptive_weights: 'Adapt-W', enable_microstructure: 'Micro',
    enable_liquidation_detector: 'Liq-Det', enable_arbitrage: 'Arb',
    enable_recovery: 'Recovery',
  };

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-3 space-y-2">
      {/* Row 1: Operational stats + module pills */}
      <div className="flex items-center gap-4 overflow-x-auto">
        <span className={`text-[10px] font-bold px-2 py-0.5 rounded shrink-0 ${
          isPaper
            ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
            : 'bg-accent-green/20 text-accent-green border border-accent-green/30'
        }`}>
          {isPaper ? 'PAPER' : 'LIVE'}
        </span>

        <div className="text-[11px] text-gray-400 shrink-0">
          <span className="text-gray-500">Up:</span>{' '}
          <span className="text-gray-300">{formatUptime(uptime)}</span>
        </div>

        <div className="text-[11px] text-gray-400 shrink-0">
          <span className="text-gray-500">Rate:</span>{' '}
          <span className="text-gray-300">{cyclesPerMin}</span>
          <span className="text-gray-600"> c/m</span>
        </div>

        <div className="text-[11px] text-gray-400 shrink-0">
          <span className="text-gray-500">Assets:</span>{' '}
          <span className="text-gray-300">{status?.product_ids?.length ?? 0}</span>
        </div>

        <div className="w-px h-4 bg-surface-3 shrink-0" />

        <div className="flex items-center gap-1.5 flex-wrap">
          {Object.entries(MODULE_LABELS).map(([key, label]) => {
            const enabled = flags[key] ?? false;
            return (
              <span
                key={key}
                className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${
                  enabled
                    ? 'bg-accent-green/10 text-accent-green border border-accent-green/20'
                    : 'bg-surface-2 text-gray-600 border border-surface-3'
                }`}
              >
                {label}
              </span>
            );
          })}
        </div>

        <div className="flex-1" />

        <div className="text-[11px] text-gray-500 shrink-0">
          WS: <span className="text-gray-400">{status?.ws_clients ?? 0}</span>
        </div>
      </div>

      {/* Row 2: Health indicators grid */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
        {indicators.map(ind => (
          <div
            key={ind.label}
            className={`flex items-center gap-1.5 px-2 py-1 rounded text-[11px] border ${
              ind.health === 'green'
                ? 'bg-accent-green/5 border-accent-green/15'
                : ind.health === 'yellow'
                ? 'bg-yellow-500/5 border-yellow-500/15'
                : 'bg-accent-red/5 border-accent-red/15'
            }`}
          >
            <HealthDot health={ind.health} />
            <span className="text-gray-500">{ind.label}:</span>
            <span className={`font-mono truncate ${
              ind.health === 'green' ? 'text-accent-green'
              : ind.health === 'yellow' ? 'text-yellow-400'
              : 'text-accent-red'
            }`}>
              {ind.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
