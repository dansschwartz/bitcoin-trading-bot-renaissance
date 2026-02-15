import type { TimeRange } from '../../types';

const RANGES: TimeRange[] = ['1H', '4H', '1D', '1W', '1M', 'ALL'];

interface Props {
  value: TimeRange;
  onChange: (range: TimeRange) => void;
}

export default function TimeRangeSelector({ value, onChange }: Props) {
  return (
    <div className="flex items-center gap-1 bg-surface-2 rounded-lg p-0.5">
      {RANGES.map((r) => (
        <button
          key={r}
          onClick={() => onChange(r)}
          className={`px-2.5 py-1 text-xs font-mono rounded-md transition-colors ${
            value === r
              ? 'bg-accent-blue text-white'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          {r}
        </button>
      ))}
    </div>
  );
}
