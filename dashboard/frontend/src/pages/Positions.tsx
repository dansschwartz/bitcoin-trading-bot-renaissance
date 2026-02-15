import PageShell from '../components/layout/PageShell';
import OpenPositionsTable from '../components/tables/OpenPositionsTable';
import ClosedPositionsTable from '../components/tables/ClosedPositionsTable';
import ExposurePanel from '../components/panels/ExposurePanel';

export default function Positions() {
  return (
    <PageShell title="Positions" subtitle="Open positions, trade history, and execution details">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="lg:col-span-2">
          <OpenPositionsTable />
        </div>
        <ExposurePanel />
      </div>

      <ClosedPositionsTable />
    </PageShell>
  );
}
