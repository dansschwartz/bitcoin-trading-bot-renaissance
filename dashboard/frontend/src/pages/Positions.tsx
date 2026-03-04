import { useState } from 'react';
import PageShell from '../components/layout/PageShell';
import OpenPositionsTable from '../components/tables/OpenPositionsTable';
import ClosedPositionsTable from '../components/tables/ClosedPositionsTable';
import ExposurePanel from '../components/panels/ExposurePanel';
import AssetSummaryPanel from '../components/panels/AssetSummaryPanel';
import PositionSummaryCards from '../components/cards/PositionSummaryCards';

const DATE_PRESETS: { label: string; value: string | undefined }[] = [
  { label: 'All Time', value: undefined },
  { label: 'Today', value: new Date().toISOString().slice(0, 10) },
  { label: 'Last 3d', value: new Date(Date.now() - 3 * 86400000).toISOString().slice(0, 10) },
  { label: 'Last 7d', value: new Date(Date.now() - 7 * 86400000).toISOString().slice(0, 10) },
  { label: 'Last 30d', value: new Date(Date.now() - 30 * 86400000).toISOString().slice(0, 10) },
];

export default function Positions() {
  const [startDate, setStartDate] = useState<string | undefined>(undefined);
  const [customDate, setCustomDate] = useState('');

  return (
    <PageShell title="Positions" subtitle="Open positions, trade history, and execution details">
      {/* Asset summary at top */}
      <AssetSummaryPanel />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="lg:col-span-2">
          <OpenPositionsTable />
        </div>
        <ExposurePanel />
      </div>

      {/* Date filter for closed positions stats */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-gray-500 font-medium">Filter closed:</span>
        <div className="flex items-center gap-1 bg-surface-2 rounded-lg p-0.5">
          {DATE_PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => { setStartDate(p.value); setCustomDate(''); }}
              className={`px-2.5 py-1 text-xs font-mono rounded-md transition-colors ${
                startDate === p.value && !customDate
                  ? 'bg-accent-blue text-white'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
        <input
          type="date"
          value={customDate}
          onChange={(e) => {
            setCustomDate(e.target.value);
            setStartDate(e.target.value || undefined);
          }}
          className="px-2 py-1 text-xs font-mono bg-surface-2 border border-surface-3 rounded-md text-gray-300 focus:outline-none focus:border-accent-blue"
          title="Custom start date"
        />
      </div>

      {/* Round-trip position summary cards */}
      <PositionSummaryCards startDate={startDate} />

      <ClosedPositionsTable startDate={startDate} />
    </PageShell>
  );
}
