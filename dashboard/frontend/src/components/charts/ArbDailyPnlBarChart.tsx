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
    const load = (raw: DailyRow[]) => {
      const byDate = new Map(raw.map((r) => [r.date, r]));
      const days: DailyRow[] = [];
      const now = new Date();
      for (let i = 30; i >= 0; i--) {
        const d = new Date(now);
        d.setDate(d.getDate() - i);
        const key = d.toISOString().slice(0, 10);
        days.push(byDate.get(key) || { date: key, pnl: 0, trades: 0, wins: 0 });
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

  const formatted = data.map((d) => {
    const dt = new Date(d.date + "T00:00:00");
    const label = dt.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    return { ...d, label };
  });

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
