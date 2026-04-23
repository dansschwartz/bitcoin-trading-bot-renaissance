import { useDashboard } from '../../context/DashboardContext';
import { formatCurrency, pnlColor, pnlSign } from '../../utils/formatters';
import MetricCard from './MetricCard';

export default function PnLCard() {
  const { state } = useDashboard();
  const pnl = state.pnl;

  if (!pnl) {
    return <MetricCard title="Total P&L" value="--" subtitle="Loading..." />;
  }

  const totalPnl = pnl.total_pnl ?? (pnl.realized_pnl + (pnl.unrealized_pnl || 0));
  const color = pnlColor(totalPnl);
  const closedCount = pnl.total_round_trips ?? pnl.total_trades ?? 0;
  const winCount = pnl.winning_round_trips ?? 0;
  const wrLabel = pnl.win_rate != null
    ? `WR: ${(pnl.win_rate * 100).toFixed(1)}% (${winCount}/${closedCount} closed)`
    : '';

  return (
    <MetricCard
      title="Total P&L"
      value={`${pnlSign(totalPnl)}${formatCurrency(totalPnl)}`}
      valueColor={color}
      subtitle={`Realized: ${pnlSign(pnl.realized_pnl)}${formatCurrency(pnl.realized_pnl)} | Unrealized: ${pnlSign(pnl.unrealized_pnl || 0)}${formatCurrency(pnl.unrealized_pnl || 0)} | ${wrLabel}`}
    />
  );
}
