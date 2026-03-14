import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { api } from "../../api";
import { formatCurrency } from "../../utils/formatters";

interface DailyRow {
  date: string;
  pnl: number;
  trades: number;
  wins: number;
}

export default function ArbDailyPnlBarChart() {
  const [data, setData] = useState<DailyRow[]>([]);

  useEffect(() => {
    api.arbDailyPnl(30).then(setData).catch(() => {});
    const id = setInterval(() => api.arbDailyPnl(30).then(setData).catch(() => {}), 60_000);
    return () => clearInterval(id);
  }, []);

  if (!data.length) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Daily P&L (30d)</h3>
        <div className="text-sm text-gray-600 py-8 text-center">No daily data yet</div>
      </div>
    );
  }

  const totalPnl = data.reduce((sum, d) => sum + d.pnl, 0);
  const totalTrades = data.reduce((sum, d) => sum + d.trades, 0);
  const profitDays = data.filter((d) => d.pnl > 0).length;

  const formatted = data.map((d) => {
    const dt = new Date(d.date + "T00:00:00");
    const label = dt.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    return { ...d, label };
  });

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Daily P&L (30d)</h3>
        <div className="flex gap-4 text-xs text-gray-500">
          <span>
            Total:{" "}
            <span className={totalPnl >= 0 ? "text-accent-green" : "text-accent-red"}>
              {totalPnl >= 0 ? "+" : ""}{formatCurrency(totalPnl)}
            </span>
          </span>
          <span>{totalTrades} trades</span>
          <span>{profitDays}/{data.length} green days</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={formatted} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
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
            contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", borderRadius: 8 }}
            labelStyle={{ color: "#9ca3af", fontSize: 12 }}
            itemStyle={{ color: "#e5e7eb", fontSize: 12 }}
            formatter={(value: number, _name: string, props: { payload?: DailyRow }) => {
              const trades = props.payload?.trades ?? 0;
              const wins = props.payload?.wins ?? 0;
              const wr = trades > 0 ? ((wins / trades) * 100).toFixed(0) : "0";
              return [
                `${formatCurrency(value)}  (${trades} trades, ${wr}% win rate)`,
                "P&L",
              ];
            }}
          />
          <ReferenceLine y={0} stroke="#374151" />
          <Bar dataKey="pnl" radius={[4, 4, 0, 0]} maxBarSize={24}>
            {formatted.map((entry, idx) => (
              <Cell key={idx} fill={entry.pnl >= 0 ? "#00d395" : "#ff4757"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
