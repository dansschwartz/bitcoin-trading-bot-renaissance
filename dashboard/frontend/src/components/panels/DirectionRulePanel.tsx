import { useEffect, useState } from 'react';
import { api } from '../../api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface RuleStat {
  rule: string;
  pair: string;
  trades: number;
  win_rate: number;
  total_pnl_bps: number;
  avg_pnl_bps: number;
}

interface AbData {
  ab_results: Record<string, RuleStat>;
}

interface AggRule {
  rule: string;
  trades: number;
  wins: number;
  winRate: number;
  totalPnlBps: number;
  avgPnlBps: number;
  pairs: number;
}

export default function DirectionRulePanel() {
  const [agg, setAgg] = useState<AggRule[]>([]);

  useEffect(() => {
    const fetch = () =>
      api.sprayAbTest().then(d => {
        const data = d as unknown as AbData;
        const results = data.ab_results || {};
        const byRule: Record<string, { trades: number; wins: number; pnl: number; pairs: Set<string> }> = {};
        for (const stat of Object.values(results)) {
          const rule = stat.rule || 'unknown';
          if (!byRule[rule]) byRule[rule] = { trades: 0, wins: 0, pnl: 0, pairs: new Set() };
          byRule[rule].trades += stat.trades;
          byRule[rule].wins += Math.round((stat.win_rate || 0) * stat.trades);
          byRule[rule].pnl += stat.total_pnl_bps || 0;
          if (stat.pair) byRule[rule].pairs.add(stat.pair);
        }
        const sorted = Object.entries(byRule)
          .map(([rule, d]) => ({
            rule,
            trades: d.trades,
            wins: d.wins,
            winRate: d.trades > 0 ? (d.wins / d.trades) * 100 : 0,
            totalPnlBps: d.pnl,
            avgPnlBps: d.trades > 0 ? d.pnl / d.trades : 0,
            pairs: d.pairs.size,
          }))
          .sort((a, b) => b.avgPnlBps - a.avgPnlBps);
        setAgg(sorted);
      }).catch(() => {});
    fetch();
    const id = setInterval(fetch, 30_000);
    return () => clearInterval(id);
  }, []);

  if (agg.length === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Direction Rule Attribution</h3>
        <div className="text-sm text-gray-600 py-4 text-center">No spray data yet</div>
      </div>
    );
  }

  const chartData = agg.map(r => ({
    name: r.rule,
    value: r.avgPnlBps,
    trades: r.trades,
  }));

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Direction Rule Attribution</h3>
        <span className="text-[10px] text-gray-600">{agg.reduce((s, r) => s + r.trades, 0)} total tokens</span>
      </div>

      <div className="h-40 mb-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 10, bottom: 0, left: 80 }}>
            <XAxis type="number" tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false}
                   tickFormatter={v => `${v}bp`} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#9ca3af' }}
                   axisLine={false} tickLine={false} width={80} />
            <Tooltip contentStyle={{
              backgroundColor: '#1a2235', border: '1px solid #243049',
              borderRadius: 8, fontSize: 11, color: '#e5e7eb',
            }} formatter={(value: number) => [`${value.toFixed(1)} bps`, 'Avg P&L']} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((d, i) => (
                <Cell key={i} fill={d.value >= 0 ? '#00d395' : '#ff4757'} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <table className="w-full text-[11px] font-mono">
        <thead>
          <tr className="text-gray-600 border-b border-surface-3">
            <th className="text-left py-1 font-medium">Rule</th>
            <th className="text-right py-1 font-medium">Tokens</th>
            <th className="text-right py-1 font-medium">Win Rate</th>
            <th className="text-right py-1 font-medium">Avg (bps)</th>
            <th className="text-right py-1 font-medium">Total (bps)</th>
            <th className="text-right py-1 font-medium">Pairs</th>
          </tr>
        </thead>
        <tbody>
          {agg.map((r, i) => (
            <tr key={i} className={`border-b border-surface-3/30 ${i === 0 ? 'bg-accent-green/5' : ''}`}>
              <td className="py-1 text-gray-300">{r.rule}{i === 0 ? ' *' : ''}</td>
              <td className="py-1 text-right text-gray-400">{r.trades.toLocaleString()}</td>
              <td className={`py-1 text-right ${r.winRate >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
                {r.winRate.toFixed(1)}%
              </td>
              <td className={`py-1 text-right ${r.avgPnlBps >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                {r.avgPnlBps >= 0 ? '+' : ''}{r.avgPnlBps.toFixed(1)}
              </td>
              <td className={`py-1 text-right ${r.totalPnlBps >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                {r.totalPnlBps >= 0 ? '+' : ''}{r.totalPnlBps.toFixed(0)}
              </td>
              <td className="py-1 text-right text-gray-500">{r.pairs}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
