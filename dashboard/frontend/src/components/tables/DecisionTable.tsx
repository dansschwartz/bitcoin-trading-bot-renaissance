import { useDashboard } from '../../context/DashboardContext';
import { formatTimestamp } from '../../utils/formatters';
import { ACTION_COLORS, regimeColor } from '../../utils/colors';

export default function DecisionTable() {
  const { state } = useDashboard();
  const decisions = state.recentDecisions;

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Recent Decisions</h3>
      <div className="overflow-x-auto max-h-96 overflow-y-auto">
        <table className="w-full text-xs font-mono">
          <thead className="sticky top-0 bg-surface-1">
            <tr className="text-gray-500 border-b border-surface-3">
              <th className="text-left py-2 px-2">Time</th>
              <th className="text-left py-2 px-2">Asset</th>
              <th className="text-left py-2 px-2">Action</th>
              <th className="text-right py-2 px-2">Confidence</th>
              <th className="text-right py-2 px-2">Signal</th>
              <th className="text-right py-2 px-2">Position</th>
              <th className="text-left py-2 px-2">Regime</th>
              <th className="text-right py-2 px-2">VAE</th>
            </tr>
          </thead>
          <tbody>
            {decisions.map((d) => (
              <tr key={d.id} className="border-b border-surface-3/50 hover:bg-surface-2/50">
                <td className="py-2 px-2 text-gray-500">{formatTimestamp(d.timestamp)}</td>
                <td className="py-2 px-2 text-gray-300">{d.product_id}</td>
                <td className="py-2 px-2">
                  <span className="font-semibold" style={{ color: ACTION_COLORS[d.action] || '#6b7280' }}>
                    {d.action}
                  </span>
                </td>
                <td className="py-2 px-2 text-right text-gray-300">{(d.confidence * 100).toFixed(1)}%</td>
                <td className="py-2 px-2 text-right text-gray-400">{d.weighted_signal.toFixed(4)}</td>
                <td className="py-2 px-2 text-right text-gray-400">{d.position_size.toFixed(4)}</td>
                <td className="py-2 px-2">
                  {d.hmm_regime && (
                    <span className="text-[10px] capitalize" style={{ color: regimeColor(d.hmm_regime) }}>
                      {d.hmm_regime}
                    </span>
                  )}
                </td>
                <td className="py-2 px-2 text-right text-gray-500">
                  {d.vae_loss != null ? d.vae_loss.toFixed(4) : '--'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
