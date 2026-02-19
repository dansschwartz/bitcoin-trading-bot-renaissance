import { useEffect, useState } from 'react';
import PageShell from '../components/layout/PageShell';
import EquityCurve from '../components/charts/EquityCurve';
import ReturnDistribution from '../components/charts/ReturnDistribution';
import CalendarHeatmap from '../components/charts/CalendarHeatmap';
import { api } from '../api';
import type { RegimePerformance, HourlyPnL } from '../types';
import { regimeColor, CHART_COLORS } from '../utils/colors';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import SignalAttributionPanel from '../components/panels/SignalAttributionPanel';

function RegimePerformanceTable() {
  const [data, setData] = useState<RegimePerformance[]>([]);

  useEffect(() => {
    api.byRegime().then(setData).catch(() => {});
  }, []);

  if (data.length === 0) return null;

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Performance by Regime</h3>
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-gray-500 border-b border-surface-3">
            <th className="text-left py-2 px-2">Regime</th>
            <th className="text-left py-2 px-2">Action</th>
            <th className="text-right py-2 px-2">Count</th>
            <th className="text-right py-2 px-2">Avg Conf</th>
            <th className="text-right py-2 px-2">Avg Signal</th>
          </tr>
        </thead>
        <tbody>
          {data.map((r, i) => (
            <tr key={i} className="border-b border-surface-3/30">
              <td className="py-2 px-2 capitalize" style={{ color: regimeColor(r.regime) }}>
                {r.regime}
              </td>
              <td className="py-2 px-2 text-gray-300">{r.action}</td>
              <td className="py-2 px-2 text-right text-gray-400">{r.count}</td>
              <td className="py-2 px-2 text-right text-gray-400">{(r.avg_confidence * 100).toFixed(1)}%</td>
              <td className="py-2 px-2 text-right text-gray-400">{r.avg_signal.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HourlyHeatmap() {
  const [data, setData] = useState<HourlyPnL[]>([]);

  useEffect(() => {
    api.hourly().then(setData).catch(() => {});
  }, []);

  if (data.length === 0) return null;

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Hourly P&L Distribution</h3>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <XAxis dataKey="hour" tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #243049', borderRadius: 8, fontSize: 12, color: '#e5e7eb' }} />
            <Bar dataKey="avg_pnl" radius={[4, 4, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.avg_pnl >= 0 ? '#00d395' : '#ff4757'} fillOpacity={0.7} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ExecutionPerformance() {
  const [data, setData] = useState<{ algo_used: string; count: number; avg_slippage: number }[]>([]);

  useEffect(() => {
    api.byExecution().then(setData).catch(() => {});
  }, []);

  if (data.length === 0) return null;

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Execution Performance</h3>
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-gray-500 border-b border-surface-3">
            <th className="text-left py-2 px-2">Algorithm</th>
            <th className="text-right py-2 px-2">Trades</th>
            <th className="text-right py-2 px-2">Avg Slippage</th>
          </tr>
        </thead>
        <tbody>
          {data.map((r, i) => (
            <tr key={i} className="border-b border-surface-3/30">
              <td className="py-2 px-2 text-gray-300">{r.algo_used || 'Unknown'}</td>
              <td className="py-2 px-2 text-right text-gray-400">{r.count}</td>
              <td className="py-2 px-2 text-right text-gray-400">
                {r.avg_slippage != null ? `${(r.avg_slippage * 100).toFixed(4)}%` : '--'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Analytics() {
  return (
    <PageShell title="Analytics" subtitle="Performance breakdown, distributions, and attribution">
      <EquityCurve />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <ReturnDistribution />
        <CalendarHeatmap />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <RegimePerformanceTable />
        <HourlyHeatmap />
      </div>

      <ExecutionPerformance />

      {/* Signal Attribution */}
      <SignalAttributionPanel />
    </PageShell>
  );
}
