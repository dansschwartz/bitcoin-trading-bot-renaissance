import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { api } from '../../api';
import { formatCurrency } from '../../utils/formatters';

const STRATEGY_KEYS = [
  "cross_exchange",
  "triangular",
  "funding_rate",
  "basis_trading",
  "listing_arb",
  "pairs_arb",
] as const;

const STRATEGY_COLORS: Record<string, string> = {
  cross_exchange: "#3b82f6",
  triangular: "#a855f7",
  funding_rate: "#00d395",
  basis_trading: "#fb923c",
  listing_arb: "#ec4899",
  pairs_arb: "#2dd4bf",
};

const STRATEGY_LABELS: Record<string, string> = {
  cross_exchange: "Cross-Exchange",
  triangular: "Triangular",
  funding_rate: "Funding Rate",
  basis_trading: "Basis",
  listing_arb: "Listing",
  pairs_arb: "Pairs",
};

interface HourlyRow {
  hour: string;
  pnl: number;
  trades: number;
  wins: number;
  by_strategy?: Record<string, number>;
}

interface ChartRow {
  hour: string;
  label: string;
  pnl: number;
  trades: number;
  wins: number;
  cross_exchange: number;
  triangular: number;
  funding_rate: number;
  basis_trading: number;
  listing_arb: number;
  pairs_arb: number;
  by_strategy?: Record<string, number>;
}

export default function ArbDailyPnlChart() {
  const [data, setData] = useState<ChartRow[]>([]);

  useEffect(() => {
    const load = (raw: HourlyRow[]) => {
      const formatted = raw.map((d) => {
        const dt = new Date(d.hour.replace(' ', 'T') + ':00');
        const h = dt.getHours();
        const ampm = h >= 12 ? 'pm' : 'am';
        const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h;
        const label = `${dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} ${h12}${ampm}`;
        const bs = d.by_strategy || {};
        return {
          ...d,
          label,
          cross_exchange: bs.cross_exchange ?? 0,
          triangular: bs.triangular ?? 0,
          funding_rate: bs.funding_rate ?? 0,
          basis_trading: bs.basis_trading ?? 0,
          listing_arb: bs.listing_arb ?? 0,
          pairs_arb: bs.pairs_arb ?? 0,
        };
      });
      setData(formatted);
    };

    api.arbHourlyPnl(48).then(load).catch(() => {});
    const id = setInterval(() => api.arbHourlyPnl(48).then(load).catch(() => {}), 30_000);
    return () => clearInterval(id);
  }, []);

  if (!data.length) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Hourly P&L (48h)</h3>
        <div className="text-sm text-gray-600 py-8 text-center">No hourly data yet</div>
      </div>
    );
  }

  // Determine which strategies have any data
  const activeStrategies = STRATEGY_KEYS.filter((k) =>
    data.some((d) => d[k] !== 0)
  );

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Hourly P&L (48h)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} stackOffset="sign" margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="label"
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            axisLine={{ stroke: '#374151' }}
            tickLine={false}
            interval="preserveStartEnd"
            angle={-35}
            textAnchor="end"
            height={50}
          />
          <YAxis
            tick={{ fill: '#6b7280', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            width={50}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0]?.payload as ChartRow | undefined;
              if (!row) return null;
              const netPnl = row.pnl;
              return (
                <div style={{
                  backgroundColor: '#111827',
                  border: '1px solid #374151',
                  borderRadius: 8,
                  padding: '8px 12px',
                  fontSize: 12,
                }}>
                  <div style={{ color: '#9ca3af', marginBottom: 4 }}>{label}</div>
                  <div style={{ color: netPnl >= 0 ? '#00d395' : '#ff4757', fontWeight: 600, marginBottom: 4 }}>
                    Net: {formatCurrency(netPnl)} ({row.trades} trades, {row.wins} wins)
                  </div>
                  {activeStrategies.map((k) => {
                    const val = row[k];
                    if (val === 0) return null;
                    return (
                      <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: 16 }}>
                        <span style={{ color: STRATEGY_COLORS[k] }}>{STRATEGY_LABELS[k]}</span>
                        <span style={{ color: val >= 0 ? '#00d395' : '#ff4757', fontFamily: 'monospace' }}>
                          {val >= 0 ? '+' : ''}{formatCurrency(val)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              );
            }}
          />
          <ReferenceLine y={0} stroke="#374151" />
          {activeStrategies.map((k) => (
            <Bar
              key={k}
              dataKey={k}
              stackId="pnl"
              fill={STRATEGY_COLORS[k]}
              radius={0}
              maxBarSize={20}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
      {/* Legend */}
      {activeStrategies.length > 1 && (
        <div className="flex flex-wrap gap-3 mt-2 justify-center">
          {activeStrategies.map((k) => (
            <div key={k} className="flex items-center gap-1 text-[10px] text-gray-400">
              <div
                className="w-2.5 h-2.5 rounded-sm"
                style={{ backgroundColor: STRATEGY_COLORS[k] }}
              />
              {STRATEGY_LABELS[k]}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
