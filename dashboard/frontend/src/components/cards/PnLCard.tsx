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
  const wr = pnl.win_rate != null ? `WR: ${(pnl.win_rate * 100).toFixed(1)}%` : '';
  const trips = pnl.total_round_trips != null ? `${pnl.total_round_trips} round-trips` : `${pnl.total_trades} trades`;

  return (
    <MetricCard
      title="Total P&L (24h)"
      value={`${pnlSign(totalPnl)}${formatCurrency(totalPnl)}`}
      valueColor={color}
      subtitle={`R: ${pnlSign(pnl.realized_pnl)}${formatCurrency(pnl.realized_pnl)} | U: ${pnlSign(pnl.unrealized_pnl || 0)}${formatCurrency(pnl.unrealized_pnl || 0)} | ${trips} | ${wr}`}
    />
  );
}
