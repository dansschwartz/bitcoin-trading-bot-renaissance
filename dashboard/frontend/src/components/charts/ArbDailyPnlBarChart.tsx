import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from "recharts";
import { api } from "../../api";
import { formatCurrency } from "../../utils/formatters";

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

interface ByStrategy {
  [key: string]: { pnl: number; trades: number; wins: number } | undefined;
}

interface DailyRow {
  date: string;
  pnl: number;
  trades: number;
  wins: number;
  by_strategy?: ByStrategy;
}

interface ChartRow {
  date: string;
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
  by_strategy?: ByStrategy;
}

export default function ArbDailyPnlBarChart() {
  const [data, setData] = useState<ChartRow[]>([]);

  useEffect(() => {
    const load = (raw: DailyRow[]) => {
      const byDate = new Map(raw.map((r) => [r.date, r]));
      const days: ChartRow[] = [];
      const now = new Date();
      for (let i = 30; i >= 0; i--) {
        const d = new Date(now);
        d.setDate(d.getDate() - i);
        const key = d.toISOString().slice(0, 10);
        const row = byDate.get(key);
        const dt = new Date(key + "T00:00:00");
        const label = dt.toLocaleDateString("en-US", { month: "short", day: "numeric" });
        const bs = row?.by_strategy || {};
        days.push({
          date: key,
          label,
          pnl: row?.pnl ?? 0,
          trades: row?.trades ?? 0,
          wins: row?.wins ?? 0,
          cross_exchange: bs.cross_exchange?.pnl ?? 0,
          triangular: bs.triangular?.pnl ?? 0,
          funding_rate: bs.funding_rate?.pnl ?? 0,
          basis_trading: bs.basis_trading?.pnl ?? 0,
          listing_arb: bs.listing_arb?.pnl ?? 0,
          pairs_arb: bs.pairs_arb?.pnl ?? 0,
          by_strategy: bs,
        });
      }
      setData(days);
    };

    api.arbDailyPnl(31).then(load).catch(() => {});
    const id = setInterval(() => api.arbDailyPnl(31).then(load).catch(() => {}), 60_000);
    return () => clearInterval(id);
  }, []);

  const totalPnl = data.reduce((sum, d) => sum + d.pnl, 0);
  const totalTrades = data.reduce((sum, d) => sum + d.trades, 0);
  const profitDays = data.filter((d) => d.pnl > 0).length;
  const activeDays = data.filter((d) => d.trades > 0).length;

  // Determine which strategies have any data
  const activeStrategies = STRATEGY_KEYS.filter((k) =>
    data.some((d) => d[k] !== 0)
  );

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Daily P&L (31d)</h3>
        <div className="flex gap-4 text-xs text-gray-500">
          <span>
            Total:{" "}
            <span className={totalPnl >= 0 ? "text-accent-green" : "text-accent-red"}>
              {totalPnl >= 0 ? "+" : ""}{formatCurrency(totalPnl)}
            </span>
          </span>
          <span>{totalTrades} trades</span>
          <span>{profitDays}/{activeDays} green days</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} stackOffset="sign" margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="label"
            tick={{ fill: "#9ca3af", fontSize: 10 }}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
            interval="preserveStartEnd"
            angle={-35}
            textAnchor="end"
            height={50}
          />
          <YAxis
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            width={55}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0]?.payload as ChartRow | undefined;
              if (!row) return null;
              const netPnl = row.pnl;
              const wr = row.trades > 0 ? ((row.wins / row.trades) * 100).toFixed(0) : "0";
              return (
                <div style={{
                  backgroundColor: "#111827",
                  border: "1px solid #374151",
                  borderRadius: 8,
                  padding: "8px 12px",
                  fontSize: 12,
                }}>
                  <div style={{ color: "#9ca3af", marginBottom: 4 }}>{label}</div>
                  <div style={{ color: netPnl >= 0 ? "#00d395" : "#ff4757", fontWeight: 600, marginBottom: 4 }}>
                    Net: {formatCurrency(netPnl)} ({row.trades} trades, {wr}% WR)
                  </div>
                  {activeStrategies.map((k) => {
                    const val = row[k];
                    if (val === 0) return null;
                    return (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
                        <span style={{ color: STRATEGY_COLORS[k] }}>{STRATEGY_LABELS[k]}</span>
                        <span style={{ color: val >= 0 ? "#00d395" : "#ff4757", fontFamily: "monospace" }}>
                          {val >= 0 ? "+" : ""}{formatCurrency(val)}
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
              maxBarSize={24}
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
