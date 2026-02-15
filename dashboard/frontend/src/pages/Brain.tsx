import PageShell from '../components/layout/PageShell';
import EnsemblePanel from '../components/panels/EnsemblePanel';
import ConfluencePanel from '../components/panels/ConfluencePanel';
import VAEGauge from '../components/panels/VAEGauge';
import RegimeCard from '../components/cards/RegimeCard';
import PredictionHistory from '../components/charts/PredictionHistory';
import RegimeTimeline from '../components/charts/RegimeTimeline';
import DecisionTable from '../components/tables/DecisionTable';
import ConditionalPanel from '../components/shared/ConditionalPanel';
import { useEffect, useState } from 'react';
import { api } from '../api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { CHART_COLORS } from '../utils/colors';

function SignalWeightsChart() {
  const [weights, setWeights] = useState<Record<string, number>>({});

  useEffect(() => {
    api.signalWeights().then(d => setWeights(d.weights || {})).catch(() => {});
  }, []);

  const data = Object.entries(weights)
    .map(([name, weight]) => ({ name, weight }))
    .sort((a, b) => b.weight - a.weight);

  if (data.length === 0) return null;

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Signal Weights</h3>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 80, right: 10, top: 5, bottom: 5 }}>
            <XAxis type="number" tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #243049', borderRadius: 8, fontSize: 12 }} />
            <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
              {data.map((_, i) => (
                <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function Brain() {
  return (
    <PageShell title="The Brain" subtitle="ML ensemble intelligence, regime classification, and decision reasoning">
      {/* Top row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <RegimeCard />
        <EnsemblePanel />
        <ConditionalPanel flag="risk_gateway" fallback={<VAEGauge />}>
          <VAEGauge />
        </ConditionalPanel>
      </div>

      {/* Regime + Confluence */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <RegimeTimeline />
        <ConfluencePanel />
      </div>

      {/* Prediction history + Signal Weights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <PredictionHistory />
        <SignalWeightsChart />
      </div>

      {/* Decision Table */}
      <DecisionTable />
    </PageShell>
  );
}
