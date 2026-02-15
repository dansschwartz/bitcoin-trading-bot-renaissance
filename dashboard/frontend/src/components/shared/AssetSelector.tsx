import { useDashboard } from '../../context/DashboardContext';

interface Props {
  value: string;
  onChange: (asset: string) => void;
}

export default function AssetSelector({ value, onChange }: Props) {
  const { state } = useDashboard();
  const assets = state.config?.product_ids ?? ['BTC-USD'];

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-surface-2 text-gray-300 text-xs font-mono rounded-lg px-3 py-1.5 border border-surface-3 focus:outline-none focus:border-accent-blue"
    >
      {assets.map((a) => (
        <option key={a} value={a}>{a}</option>
      ))}
    </select>
  );
}
