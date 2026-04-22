import { useEffect, useState } from 'react';
import { api } from '../../api';

interface ModelStat {
  model: string;
  total_evaluated: number;
  correct: number;
  wrong: number;
  skipped: number;
  accuracy: number;
  avg_confidence: number;
  avg_correct_confidence: number;
}

interface AccuracyData {
  window_hours: number;
  total_predictions: number;
  evaluated: number;
  overall_accuracy: number;
  models: ModelStat[];
}

function AccuracyBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 55 ? 'bg-accent-green' : pct >= 50 ? 'bg-yellow-500' : 'bg-accent-red';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-surface-3 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-[11px] font-mono w-10 text-right ${pct >= 55 ? 'text-accent-green' : pct >= 50 ? 'text-yellow-400' : 'text-accent-red'}`}>
        {pct}%
      </span>
    </div>
  );
}

export default function ModelAccuracyPanel() {
  const [data, setData] = useState<AccuracyData | null>(null);

  useEffect(() => {
    const fetch = () =>
      api.modelAccuracy(24).then(d => setData(d as unknown as AccuracyData)).catch(() => {});
    fetch();
    const id = setInterval(fetch, 60_000);
    return () => clearInterval(id);
  }, []);

  if (!data || data.total_predictions === 0) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Model Accuracy (Live)</h3>
        <div className="text-sm text-gray-600 py-4 text-center">No predictions to evaluate yet</div>
      </div>
    );
  }

  const overallPct = Math.round(data.overall_accuracy * 100);
  const overallColor = overallPct >= 55 ? 'text-accent-green' : overallPct >= 50 ? 'text-yellow-400' : 'text-accent-red';

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Model Accuracy (Live)</h3>
        <div className="flex items-center gap-3">
          <span className={`text-lg font-mono font-semibold ${overallColor}`}>
            {overallPct}%
          </span>
          <span className="text-[10px] text-gray-600">
            {data.evaluated} evaluated / {data.total_predictions} predictions (24h)
          </span>
        </div>
      </div>

      {/* Per-model accuracy table */}
      <div className="max-h-64 overflow-y-auto">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-surface-1">
            <tr className="text-gray-600 border-b border-surface-3">
              <th className="text-left py-1 font-medium">Model</th>
              <th className="text-right py-1 font-medium w-14">Eval</th>
              <th className="text-right py-1 font-medium w-10">W</th>
              <th className="text-right py-1 font-medium w-10">L</th>
              <th className="py-1 font-medium w-32 text-right">Accuracy</th>
              <th className="text-right py-1 font-medium w-14">Conf</th>
            </tr>
          </thead>
          <tbody>
            {data.models.map((m, i) => (
              <tr key={i} className="border-b border-surface-3/30">
                <td className="py-1.5 text-gray-300 font-mono">{m.model}</td>
                <td className="py-1.5 text-right text-gray-400 font-mono">{m.total_evaluated}</td>
                <td className="py-1.5 text-right text-accent-green font-mono">{m.correct}</td>
                <td className="py-1.5 text-right text-accent-red font-mono">{m.wrong}</td>
                <td className="py-1.5 pl-4">
                  <AccuracyBar value={m.accuracy} />
                </td>
                <td className="py-1.5 text-right text-gray-500 font-mono">
                  {(m.avg_confidence * 100).toFixed(0)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-2 text-[10px] text-gray-600">
        Direction accuracy: predicted sign vs actual 5-min price move. &gt;55% = profitable edge.
      </div>
    </div>
  );
}
