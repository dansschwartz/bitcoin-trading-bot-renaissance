import { useDashboard } from '../../context/DashboardContext';
import { formatCurrency, pnlColor, pnlSign } from '../../utils/formatters';
import MetricCard from './MetricCard';

export default function PnLCard() {
  const { state } = useDashboard();
  const pnl = state.pnl;

  if (!pnl) {
    return <MetricCard title="Realized P&L" value="--" subtitle="Loading..." />;
  }

  const color = pnlColor(pnl.realized_pnl);

  return (
    <MetricCard
      title="Realized P&L (24h)"
      value={`${pnlSign(pnl.realized_pnl)}${formatCurrency(pnl.realized_pnl)}`}
      valueColor={color}
      subtitle={`${pnl.total_trades} trades | WR: ${(pnl.win_rate * 100).toFixed(1)}%`}
    />
  );
}
